# main.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import feedparser
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import plotly.express as px
import io
import time
import random

st.set_page_config(page_title="News + Price Forecast", layout="wide")

st.title("📈 Stock Price Forecasting using News Sentiment")
st.markdown("Enter a ticker symbol, optionally provide a NewsAPI key, then click `Run`.")

# ----------------------------------------------------
# FIXED & IMPROVED NEWS FETCHER
# ----------------------------------------------------
def get_news(query, api_key=None, max_articles=20):
    """Fetch news from NewsAPI (if key provided) or fallback Google News RSS with retry."""
    articles = []

    # ------------------------------------
    # 1) Try NewsAPI first
    # ------------------------------------
    if api_key:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_articles,
            "apiKey": api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            data = r.json()
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "url": a.get("url", ""),
                    "publishedAt": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                })
        except Exception as e:
            st.warning(f"NewsAPI failed: {e}. Using RSS fallback.")

    if articles:
        df = pd.DataFrame(articles)
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df["publishedAt"] = df["publishedAt"].fillna(pd.Timestamp.now())
        return df

    # ------------------------------------
    # 2) RSS fallback
    # ------------------------------------
    import urllib.parse
    encoded = urllib.parse.quote(query)

    rss_urls = [
        f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en",
        f"https://news.google.com/rss/search?q={encoded}",
    ]

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64)",
    ]

    feed_entries = []

    for url in rss_urls:
        for _ in range(3):  # retry
            try:
                headers = {"User-Agent": random.choice(user_agents)}
                resp = requests.get(url, timeout=10, headers=headers)
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)

                if feed.entries:
                    feed_entries = feed.entries
                    break
            except:
                time.sleep(1)

        if feed_entries:
            break

    if not feed_entries:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["title", "description", "url", "publishedAt", "source"])

    # Parse RSS results
    for entry in feed_entries[:max_articles]:
        published = entry.get("published") or entry.get("updated") or datetime.now().isoformat()
        desc = entry.get("summary") or entry.get("description") or ""

        articles.append({
            "title": entry.get("title", ""),
            "description": desc,
            "url": entry.get("link", ""),
            "publishedAt": published,
            "source": "Google News",
        })

    df = pd.DataFrame(articles)

    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df["publishedAt"] = df["publishedAt"].fillna(pd.Timestamp.now())

    return df


# ----------------------------------------------------
# Sentiment computation
# ----------------------------------------------------
def compute_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()

    def score_text(row):
        text = ""
        if pd.notna(row.get("title")):
            text += str(row.get("title")) + ". "
        if pd.notna(row.get("description")):
            text += str(row.get("description"))
        if not text:
            return 0
        return analyzer.polarity_scores(text)["compound"]

    if df is None or df.empty:
        # Return empty DataFrame with sentiment column
        df = pd.DataFrame(columns=["title", "description", "url", "publishedAt", "source", "sentiment", "date"])
        return df

    df = df.copy()
    df["sentiment"] = df.apply(score_text, axis=1)
    df["date"] = df["publishedAt"].dt.date

    return df


# ----------------------------------------------------
# Price downloader
# ----------------------------------------------------
def fetch_price_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df

    df = df.reset_index()

    # Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[1] for c in df.columns]

    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ----------------------------------------------------
# Feature builder
# ----------------------------------------------------
def build_features(price_df, news_df, lookback=5):
    df = price_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["vol_5"] = df["Close"].pct_change().rolling(5).std()

    # sentiment aggregation
    if news_df is not None and not news_df.empty:
        news_agg = (
            news_df.groupby("date")["sentiment"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"date": "Date", "mean": "sent_mean", "count": "sent_count"})
        )
        news_agg["Date"] = pd.to_datetime(news_agg["Date"])
        df = df.merge(news_agg, on="Date", how="left")
    else:
        df["sent_mean"] = 0
        df["sent_count"] = 0

    df["sent_mean"] = df["sent_mean"].fillna(method="ffill").fillna(0)
    df["sent_count"] = df["sent_count"].fillna(0)

    # add lags
    for lag in range(1, lookback + 1):
        df[f"lag_close_{lag}"] = df["Close"].shift(lag)
        df[f"lag_return_{lag}"] = df["return_1d"].shift(lag)

    df["target_next_close"] = df["Close"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------------------------------
# UI
# ----------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Ticker (yfinance)", value="AAPL")
    company_name = st.text_input("Company/Query", value="Apple")
    start_date = st.date_input("Start date", value=(datetime.now() - timedelta(days=365)).date())
    end_date = st.date_input("End date", value=datetime.now().date())

with col2:
    news_api_key = st.text_input("NewsAPI Key (optional)", type="password")
    max_news = st.slider("Max news articles", 5, 50, 20)

# ----------------------------------------------------
# RUN BUTTON
# ----------------------------------------------------
if st.button("Run analysis"):
    with st.spinner("Fetching & computing..."):
        # 1) Price data
        price_df = fetch_price_data(ticker, start_date, end_date + timedelta(days=1))
        if price_df.empty:
            st.error("No price data found.")
            st.stop()

        # 2) News + sentiment
        news_df = get_news(company_name or ticker, api_key=news_api_key, max_articles=max_news)
        news_df = compute_sentiment(news_df)

        st.subheader("📄 News articles")
        
        # Check if news_df has data and the required columns
        if news_df.empty or len(news_df) == 0:
            st.info("No news articles found. The model will run without sentiment data.")
            # Create a minimal display
            st.dataframe(pd.DataFrame({"message": ["No articles found"]}))
        else:
            # Safely display available columns
            display_cols = []
            for col in ["publishedAt", "title", "source", "sentiment"]:
                if col in news_df.columns:
                    display_cols.append(col)
            
            if display_cols:
                display_df = news_df[display_cols].copy()
                if "publishedAt" in display_df.columns:
                    display_df = display_df.sort_values("publishedAt", ascending=False)
                st.dataframe(display_df)
            else:
                st.warning("News data retrieved but missing expected columns.")
                st.dataframe(news_df.head())

        # 3) Features
        features = build_features(price_df, news_df, lookback=5)
        if features.empty:
            st.error("Insufficient data after feature engineering.")
            st.stop()

        # Prepare X and y
        cols_to_drop = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "target_next_close"]
        X = features.drop(columns=[c for c in cols_to_drop if c in features.columns], errors="ignore")
        y = features["target_next_close"]

        # 4) Model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = math.sqrt(mean_squared_error(y_test, preds))

        st.subheader("📊 Model Performance")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("MAE", f"{mae:.4f}")
        col_m2.metric("RMSE", f"{rmse:.4f}")

        # 5) Next-day forecast
        latest = X.iloc[[-1]]
        next_price = model.predict(latest)[0]

        st.subheader("📈 Next-Day Forecast")
        st.metric(f"Predicted next close for {ticker}", f"${next_price:.2f}")

        # 6) Plot Actual vs Predicted
        test_plot = pd.DataFrame({
            "Date": features.loc[X_test.index, "Date"].values,
            "Actual": y_test.values,
            "Predicted": preds
        })

        fig = px.line(test_plot, x="Date", y=["Actual", "Predicted"], 
                     title="Actual vs Predicted Prices",
                     labels={"value": "Price", "variable": "Type"})
        st.plotly_chart(fig, use_container_width=True)

        # 7) Price chart
        fig2 = px.line(price_df, x="Date", y="Close", 
                      title=f"{ticker} Historical Close Price")
        st.plotly_chart(fig2, use_container_width=True)

        # 8) Downloads
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📥 Download News CSV", 
                             news_df.to_csv(index=False), 
                             f"{ticker}_news.csv",
                             mime="text/csv")
        with col_d2:
            st.download_button("📥 Download Price CSV", 
                             price_df.to_csv(index=False), 
                             f"{ticker}_prices.csv",
                             mime="text/csv")

        st.success("✅ Analysis complete!")

st.markdown("---")
st.caption("This app forecasts the next closing price using RandomForest + news sentiment analysis.")
