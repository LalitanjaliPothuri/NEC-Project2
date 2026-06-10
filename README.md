Live render link : https://nec-project2-2.onrender.com

# Intelligent Sales

Live Application: http://localhost:8501

A Streamlit-based sales intelligence system for forecasting demand, optimizing inventory, and generating business reports.

## Features

- Data upload for CSV and Excel files
- Data preprocessing and storage of raw/clean datasets
- EDA with sales, revenue, product, and correlation charts
- ML model training for monthly demand forecasting
- Sales forecasting with actual vs predicted analysis
- Inventory optimization with reorder points and stockout estimates
- Monthly revenue, yearly revenue, profit, and loss reporting
- Downloadable reports and dashboards

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app (project workspace):

```bash
cd "C:\\Users\\dell\\Documents\\ST-DA-Projects\\NEC_PROJECT2"
python -m streamlit run app.py --server.port 8501
```

If you are running the cloned repository version (separate folder `nec-task-2-intelligent-sales`), start it with:

```bash
cd "C:\\Users\\dell\\Documents\\ST-DA-Projects\\nec-task-2-intelligent-sales"
python -m streamlit run app.py --server.port 8502
```

Open the app in your browser at:

- Local (original workspace): http://localhost:8501
- Cloned repo: http://localhost:8502

## Notes

- The app uses demo datasets in `data/raw`. I reduced the sample dataset and added cosmetic products (Lipstick, Foundation, Moisturizer, Perfume, Nail Polish, Face Wash) for quick testing. The cloned repo also contains a trimmed demo dataset.
- Uploaded files are stored in `data/raw`.
- Cleaned outputs are saved in `data/processed`.
- Forecast and inventory reports are saved in `reports`.
- Trained model artifacts are saved in `models`.
