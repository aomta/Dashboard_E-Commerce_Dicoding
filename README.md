# Proyek Analisis Data: E-Commerce Public Dataset
Analisis data E-Commerce Public Dataset (Olist) untuk menjawab dua pertanyaan bisnis:

1. Berapa persentase pelanggan dan kontribusi pendapatan setiap segmen RFM pada Januari 2017 - Agustus 2018?
2. State mana saja yang masuk 5 besar total pendapatan pada Januari - Agustus 2018 dan berapa kontribusinya?

Teknik analisis lanjutan: RFM analysis, geospatial analysis (peta bubble), dan binning/manual grouping (tier pasar).

## Struktur Direktori

```
Dashboard_E-Commerce_Dicoding
├───dashboard
│   ├───main_data.csv
│   ├───rfm_data.csv
|   ├───state_revenue_data.csv
│   └───dashboard.py

```

## Setup Environment

```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Menjalankan Dashboard

Jalankan dari folder utama proyek:

```
streamlit run dashboard/dashboard.py
```
