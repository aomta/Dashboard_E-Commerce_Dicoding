from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------ Konfigurasi
st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")

BASE_DIR = Path(__file__).parent
HIGHLIGHT, ACCENT, MUTED = "#1f6f8b", "#e0a030", "#c9ced6"
SEGMENTS = ["Best Customers", "Loyal Customers", "Big Spenders", "Recent Customers", "Lost Customers"]
TIER_LABELS = ["Tier 3 - Potensi Kecil", "Tier 2 - Potensi Menengah", "Tier 1 - Pasar Utama"]
TIER_COLORS = {"Tier 1 - Pasar Utama": HIGHLIGHT, "Tier 2 - Potensi Menengah": ACCENT, "Tier 3 - Potensi Kecil": MUTED}


def clean_axes(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------------ Data
@st.cache_data
def load_orders():
    df = pd.read_csv(
        BASE_DIR / "main_data.csv",
        usecols=["order_id", "customer_unique_id", "customer_state", "order_purchase_timestamp", "order_value"],
        parse_dates=["order_purchase_timestamp"],
    )
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m")
    return df


@st.cache_data
def load_state_geo():
    return pd.read_csv(BASE_DIR / "state_revenue_data.csv", usecols=["customer_state", "lat", "lng"])


def quartile_score(series, labels):
    """Skor kuartil; jika batas kuartil kembar (data sempit), pakai peringkat."""
    try:
        return pd.qcut(series, q=4, labels=labels).astype(int)
    except ValueError:
        return pd.qcut(series.rank(method="first"), q=4, labels=labels).astype(int)


@st.cache_data
def compute_rfm(start, end):
    orders = load_orders()
    df = orders[(orders["order_purchase_timestamp"] >= pd.Timestamp(start))
                & (orders["order_purchase_timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ref_date = df["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id", as_index=False).agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_value", "sum"),
    )
    rfm["recency"] = (ref_date - rfm["last_purchase"]).dt.days
    if len(rfm) < 4:
        return pd.DataFrame(), pd.DataFrame()

    rfm["r_score"] = quartile_score(rfm["recency"], [4, 3, 2, 1])
    rfm["f_score"] = pd.cut(rfm["frequency"], bins=[0, 1, 2, np.inf], labels=[1, 2, 3]).astype(int)
    rfm["m_score"] = quartile_score(rfm["monetary"], [1, 2, 3, 4])

    conditions = [
        (rfm["f_score"] >= 2) & (rfm["r_score"] >= 3) & (rfm["m_score"] >= 3),
        (rfm["f_score"] >= 2),
        (rfm["f_score"] == 1) & (rfm["m_score"] == 4),
        (rfm["f_score"] == 1) & (rfm["r_score"] >= 3),
    ]
    rfm["segment"] = np.select(conditions, SEGMENTS[:4], default="Lost Customers")

    summary = rfm.groupby("segment").agg(
        total_customers=("customer_unique_id", "nunique"),
        total_revenue=("monetary", "sum"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).reindex(SEGMENTS).fillna(0).reset_index()
    summary["customer_share_pct"] = summary["total_customers"] / summary["total_customers"].sum() * 100
    summary["revenue_share_pct"] = summary["total_revenue"] / summary["total_revenue"].sum() * 100
    return rfm, summary


@st.cache_data
def compute_state_revenue(years):
    orders = load_orders()
    df = orders[orders["purchase_year"].isin(years)]
    state = (
        df.groupby("customer_state")
        .agg(total_orders=("order_id", "nunique"),
             total_customers=("customer_unique_id", "nunique"),
             total_revenue=("order_value", "sum"))
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )
    state["revenue_share_pct"] = state["total_revenue"] / state["total_revenue"].sum() * 100
    state["cumulative_share_pct"] = state["revenue_share_pct"].cumsum()
    state["avg_order_value"] = state["total_revenue"] / state["total_orders"]
    state["market_tier"] = pd.cut(state["revenue_share_pct"], bins=[0, 2, 10, 101],
                                  labels=TIER_LABELS, right=False)
    return state.merge(load_state_geo(), on="customer_state", how="left")


def rupiah(x):
    return f"R$ {x:,.0f}"


def juta(x):
    return f"R$ {x / 1e6:,.2f} jt"


# ------------------------------------------------------------------ Muat data
try:
    orders_all = load_orders()
except FileNotFoundError:
    st.error("File `main_data.csv` tidak ditemukan. Jalankan notebook sampai sel penyimpanan data dashboard, "
             "lalu letakkan `main_data.csv` dan `state_revenue_data.csv` di folder `dashboard/`.")
    st.stop()

min_date = orders_all["order_purchase_timestamp"].min().date()
max_date = orders_all["order_purchase_timestamp"].max().date()

# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("Filter")
    date_range = st.date_input("Periode transaksi (Ringkasan dan RFM)", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    st.caption("Filter ini berlaku untuk tab Ringkasan dan Segmentasi RFM. "
               "Tab Pendapatan per State memiliki pilihan periode sendiri.")
    st.divider()
    st.caption(f"Data transaksi: {min_date:%d %b %Y} s.d. {max_date:%d %b %Y}. "
               "Hanya pesanan berstatus *delivered*.")

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    st.info("Pilih tanggal awal dan tanggal akhir pada filter di sidebar.")
    st.stop()

period_df = orders_all[(orders_all["order_purchase_timestamp"] >= pd.Timestamp(start_date))
                       & (orders_all["order_purchase_timestamp"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))]

# ------------------------------------------------------------------ Header
st.title("🛒 Dashboard Analisis E-Commerce")
st.caption("Menjawab dua pertanyaan bisnis: segmentasi pelanggan (RFM) dan persebaran pendapatan per state.")

tab_overview, tab_rfm, tab_state = st.tabs(["Ringkasan", "Pertanyaan 1: Segmentasi RFM", "Pertanyaan 2: Pendapatan per State"])

# ------------------------------------------------------------------ Tab 1: Ringkasan
with tab_overview:
    if period_df.empty:
        st.warning("Tidak ada transaksi pada periode yang dipilih.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Pendapatan", juta(period_df["order_value"].sum()))
        c2.metric("Total Pesanan", f"{period_df['order_id'].nunique():,}")
        c3.metric("Total Pelanggan ", f"{period_df['customer_unique_id'].nunique():,}")
        c4.metric("Rata-rata Nilai Pesanan", rupiah(period_df["order_value"].mean()))

        monthly = period_df.groupby("purchase_month").agg(
            total_orders=("order_id", "nunique"), total_revenue=("order_value", "sum")).reset_index()

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(monthly["purchase_month"], monthly["total_revenue"] / 1e6, color=HIGHLIGHT, marker="o", linewidth=2)
            ax.set_title("Pendapatan per Bulan", loc="left", fontsize=13)
            ax.set_ylabel("Pendapatan (juta R$)")
            ax.tick_params(axis="x", rotation=60)
            ax.set_ylim(bottom=0)
            clean_axes(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(monthly["purchase_month"], monthly["total_orders"], color=MUTED)
            peak = monthly["total_orders"].idxmax()
            ax.bar(monthly.loc[peak, "purchase_month"], monthly.loc[peak, "total_orders"], color=HIGHLIGHT)
            ax.set_title("Jumlah Pesanan per Bulan (puncak disorot)", loc="left", fontsize=13)
            ax.set_ylabel("Jumlah pesanan")
            ax.tick_params(axis="x", rotation=60)
            clean_axes(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(f"Puncak pesanan terjadi pada **{monthly.loc[peak, 'purchase_month']}** "
                   f"dengan {monthly.loc[peak, 'total_orders']:,} pesanan.")

# ------------------------------------------------------------------ Tab 2: RFM
with tab_rfm:
    st.subheader("Berapa porsi pelanggan dan pendapatan tiap segmen RFM?")
    rfm_df, seg = compute_rfm(start_date, end_date)

    if seg.empty:
        st.warning("Data pada periode ini terlalu sedikit untuk membentuk segmentasi RFM.")
    else:
        seg = seg.set_index("segment")
        best, big = seg.loc["Best Customers"], seg.loc["Big Spenders"]
        repeat_pct = (rfm_df["frequency"] > 1).mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pelanggan Unik", f"{len(rfm_df):,}")
        c2.metric("Pelanggan Repeat", f"{repeat_pct:.1f}%")
        c3.metric("Best Customers", f"{best['customer_share_pct']:.2f}% pelanggan",
                  f"{best['revenue_share_pct']:.2f}% pendapatan", delta_color="off")
        c4.metric("Big Spenders", f"{big['customer_share_pct']:.2f}% pelanggan",
                  f"{big['revenue_share_pct']:.2f}% pendapatan", delta_color="off")

        plot_df = seg.reset_index().sort_values("revenue_share_pct", ascending=False)
        color_map = {"Best Customers": HIGHLIGHT, "Big Spenders": ACCENT}
        colors = [color_map.get(s, MUTED) for s in plot_df["segment"]]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, col, title in zip(axes, ["customer_share_pct", "revenue_share_pct"],
                                  ["Persentase Pelanggan per Segmen", "Persentase Pendapatan per Segmen"]):
            bars = ax.barh(plot_df["segment"], plot_df[col], color=colors)
            ax.bar_label(bars, fmt="%.1f%%", padding=4)
            ax.set_title(title, fontsize=13, loc="left")
            ax.set_xlabel("Persentase (%)")
            ax.set_xlim(0, max(plot_df[col].max() * 1.2, 1))
            clean_axes(ax)
        axes[0].invert_yaxis()
        fig.suptitle(f"Segmentasi RFM Pelanggan, {start_date:%b %Y} - {end_date:%b %Y}", fontsize=15, x=0.01, ha="left")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"**Insight:** *Best Customers* hanya **{best['customer_share_pct']:.2f}%** pelanggan dan "
            f"**{best['revenue_share_pct']:.2f}%** pendapatan, sedangkan *Big Spenders* "
            f"({big['customer_share_pct']:.2f}% pelanggan) menyumbang **{big['revenue_share_pct']:.2f}%** pendapatan "
            f"dengan rata-rata *recency* {big['avg_recency']:.0f} hari. Peluang terbesar ada pada mengubah "
            f"*Big Spenders* menjadi pembeli berulang."
        )

        table = seg.reset_index()[["segment", "total_customers", "customer_share_pct", "total_revenue",
                                   "revenue_share_pct", "avg_recency", "avg_frequency", "avg_monetary"]]
        table.columns = ["Segmen", "Pelanggan", "% Pelanggan", "Pendapatan (R$)", "% Pendapatan",
                         "Rata-rata Recency (hari)", "Rata-rata Frequency", "Rata-rata Monetary (R$)"]
        st.dataframe(
            table.style.format({"Pelanggan": "{:,.0f}", "% Pelanggan": "{:.2f}", "Pendapatan (R$)": "{:,.0f}",
                                "% Pendapatan": "{:.2f}", "Rata-rata Recency (hari)": "{:.0f}",
                                "Rata-rata Frequency": "{:.2f}", "Rata-rata Monetary (R$)": "{:,.2f}"}),
            hide_index=True,
        )

        with st.expander("Definisi segmen"):
            st.markdown(
                "- **Recency**: hari sejak pembelian terakhir (skor 4 = paling baru, kuartil).\n"
                "- **Frequency**: jumlah pesanan (skor 1 = 1x, 2 = 2x, 3 = 3x atau lebih).\n"
                "- **Monetary**: total pengeluaran (skor 4 = kuartil tertinggi).\n"
                "- **Best Customers**: Frequency ≥ 2, Recency skor ≥ 3, Monetary skor ≥ 3.\n"
                "- **Loyal Customers**: Frequency ≥ 2 selain Best Customers.\n"
                "- **Big Spenders**: membeli 1x dengan Monetary skor 4.\n"
                "- **Recent Customers**: membeli 1x, Recency skor ≥ 3.\n"
                "- **Lost Customers**: membeli 1x, Recency skor ≤ 2 dan Monetary skor < 4."
            )

# ------------------------------------------------------------------ Tab 3: State
with tab_state:
    st.subheader("State mana yang menyumbang pendapatan terbesar?")
    col_p, col_n = st.columns([2, 1])
    with col_p:
        period_choice = st.radio("Periode", ["2018", "2017", "2017-2018"], horizontal=True)
    with col_n:
        top_n = st.slider("Jumlah state teratas yang disorot", 3, 10, 5)

    years = {"2018": (2018,), "2017": (2017,), "2017-2018": (2017, 2018)}[period_choice]
    state_df = compute_state_revenue(years)

    if state_df.empty:
        st.warning("Tidak ada data untuk periode ini.")
    else:
        span = orders_all[orders_all["purchase_year"].isin(years)]["order_purchase_timestamp"]
        top = state_df.head(top_n)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Pendapatan", juta(state_df["total_revenue"].sum()))
        c2.metric(f"Kontribusi {top_n} State Teratas", f"{top['revenue_share_pct'].sum():.2f}%")
        c3.metric("State Peringkat 1", f"{state_df.loc[0, 'customer_state']} ({state_df.loc[0, 'revenue_share_pct']:.1f}%)")
        st.caption(f"Cakupan data periode ini: {span.min():%d %b %Y} s.d. {span.max():%d %b %Y}.")

        top10 = state_df.head(10).reset_index(drop=True)
        bar_colors = [HIGHLIGHT if i < top_n else MUTED for i in range(len(top10))]
        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.barh(top10["customer_state"], top10["total_revenue"] / 1e6, color=bar_colors)
        ax.invert_yaxis()
        labels = [f"R$ {r / 1e6:.2f} jt ({s:.1f}%)" for r, s in zip(top10["total_revenue"], top10["revenue_share_pct"])]
        ax.bar_label(bars, labels=labels, padding=5)
        ax.set_xlim(0, top10["total_revenue"].max() / 1e6 * 1.3)
        ax.set_xlabel("Total Pendapatan (juta R$)")
        ax.set_ylabel("State")
        ax.set_title(f"Top 10 State berdasarkan Pendapatan {period_choice} ({top_n} teratas disorot)", fontsize=14, loc="left")
        clean_axes(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"**Insight:** {top_n} state teratas ({', '.join(top['customer_state'])}) menyumbang "
            f"**{top['revenue_share_pct'].sum():.2f}%** pendapatan. "
            f"{state_df.loc[0, 'customer_state']} memimpin dengan {juta(state_df.loc[0, 'total_revenue'])}."
        )

        st.markdown("#### Analisis lanjutan: peta pendapatan dan tier pasar")
        map_df = state_df.dropna(subset=["lat", "lng"])
        col_map, col_tier = st.columns([3, 2])
        with col_map:
            fig, ax = plt.subplots(figsize=(8, 8))
            for tier, color in TIER_COLORS.items():
                sub = map_df[map_df["market_tier"] == tier]
                ax.scatter(sub["lng"], sub["lat"], s=sub["total_revenue"] / 1e6 * 80 + 40, color=color,
                           alpha=0.85, edgecolor="white", label=tier)
            for _, row in map_df[map_df["revenue_share_pct"] >= 2].iterrows():
                ax.annotate(row["customer_state"], (row["lng"], row["lat"]), textcoords="offset points",
                            xytext=(6, 6), fontsize=10, fontweight="bold")
            ax.set_title("Peta Bubble Pendapatan per State (ukuran = total pendapatan)", fontsize=13, loc="left")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_aspect("equal")
            ax.legend(title="Tier pasar", loc="lower left", frameon=False)
            clean_axes(ax)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col_tier:
            tier_summary = (
                state_df.groupby("market_tier", observed=True)
                .agg(total_states=("customer_state", "nunique"), revenue_share_pct=("revenue_share_pct", "sum"))
                .reset_index().sort_values("revenue_share_pct", ascending=False)
            )
            tier_summary.columns = ["Tier pasar", "Jumlah state", "% Pendapatan"]
            st.dataframe(tier_summary.style.format({"% Pendapatan": "{:.2f}"}), hide_index=True)
            st.caption("Tier ditentukan dengan binning manual dari kontribusi pendapatan: "
                       "Tier 1 ≥ 10%, Tier 2 2% sampai < 10%, Tier 3 < 2%.")

        with st.expander("Lihat tabel lengkap seluruh state"):
            full = state_df[["customer_state", "total_orders", "total_customers", "total_revenue",
                             "revenue_share_pct", "cumulative_share_pct", "avg_order_value", "market_tier"]].copy()
            full.columns = ["State", "Pesanan", "Pelanggan", "Pendapatan (R$)", "% Pendapatan",
                            "% Kumulatif", "Rata-rata Nilai Pesanan (R$)", "Tier"]
            st.dataframe(
                full.style.format({"Pesanan": "{:,.0f}", "Pelanggan": "{:,.0f}", "Pendapatan (R$)": "{:,.0f}",
                                   "% Pendapatan": "{:.2f}", "% Kumulatif": "{:.2f}",
                                   "Rata-rata Nilai Pesanan (R$)": "{:,.2f}"}),
                hide_index=True,
            )

st.caption("Proyek Analisis Data - Mutia Aulia")
