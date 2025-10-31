# StockPortfolioApp_v2 (Simple Local Save)

This simplified version uses Firebase Authentication (REST) for Signup/Login/Password Reset
and saves portfolio data locally per user in `user_data/data_{uid}.json`.

**Before running:** update `firebase_config.py` with your project's keys if needed.

Run locally:
```
pip install -r requirements.txt
streamlit run app.py
```

Note: This version saves data locally (per machine). For cloud persistence, Firestore integration is recommended.
