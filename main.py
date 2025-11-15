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

st.set_page_config(page_title="News + Price Forecast", layout="wide")

st.title("📈 Stock Price Forecasting using News Sentiment")
st.markdown("Enter a ticker symbol, optionally provide a NewsAPI key (recommended), then click `Run`.")

# ----------------------- Helpers -----------------------

def get_news(query, api_key=None, max_articles=20):
    """Fetch news either via NewsAPI (if key provided) or from Google News RSS as fallback."""
    articles = []
    if api_key:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": max_articles,
            "page": 1,
            "apiKey": api_key,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            for a in data.get("articles", []):
                articles.append({
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "url": a.get("url"),
                    "publishedAt": a.get("publishedAt"),
                    "source": a.get("source", {}).get("name"),
                })
        except Exception as e:
            st.warning(f"NewsAPI request failed: {e} — falling back to RSS.")

    if not articles:
        # Fallback: Google News RSS (no API key required)
        try:
            import urllib.parse
            q = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
            
            # Use requests to fetch RSS content first
            response = requests.get(rss_url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            # Parse the RSS feed
            feed = feedparser.parse(response.content)
            
            if feed.entries:
                for entry in feed.entries[:max_articles]:
                    # Get published date
                    published = None
                    if hasattr(entry, 'published'):
                        published = entry.published
                    elif hasattr(entry, 'updated'):
                        published = entry.updated
                    
                    # Get description/summary
                    description = None
                    if hasattr(entry, 'summary'):
                        description = entry.summary
                    elif hasattr(entry, 'description'):
                        description = entry.description
                    
                    articles.append({
                        "title": entry.get("title", ""),
                        "description": description,
                        "url": entry.get("link", ""),
                        "publishedAt": published,
                        "source": "Google News",
                    })
            else:
                st.info(f"No RSS feed entries found for '{query}'. Try a different search term.")
        except Exception as e:
            st.error(f"Failed to fetch RSS news: {str(e)}")

    # Normalize publishedAt to datetime
    for a in articles:
        if a.get("publishedAt"):
            try:
                a["publishedAt"] = pd.to_datetime(a["publishedAt"], utc=True).tz_convert(None)
            except Exception:
                try:
                    a["publishedAt"] = pd.to_datetime(a["publishedAt"])
                except Exception:
                    a["publishedAt"] = pd.Timestamp.now()
        else:
            a["publishedAt"] = pd.Timestamp.now()

    return pd.DataFrame(articles)


def compute_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    def score_text(row):
        text = ""
        if pd.notna(row.get("title")):
            text += str(row.get("title")) + ". "
        if pd.notna(row.get("description")):
            text += str(row.get("description"))
        if not text:
            return 0.0
        return analyzer.polarity_scores(text)["compound"]

    if df is None or df.empty:
        return df
    df = df.copy()
    df["sentiment"] = df.apply(score_text, axis=1)
    df["date"] = df["publishedAt"].dt.date
    return df


def fetch_price_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    # Flatten column names if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def build_features(price_df, news_df, lookback=5):
    df = price_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    # basic technical features
    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["vol_5"] = df["Close"].pct_change().rolling(5).std()

    # aggregate sentiment per date
    if news_df is not None and not news_df.empty:
        news_agg = news_df.groupby("date")["sentiment"].agg(["mean", "count"]).reset_index().rename(columns={"date": "Date", "mean": "sent_mean", "count": "sent_count"})
        news_agg["Date"] = pd.to_datetime(news_agg["Date"])
        df = df.merge(news_agg, how="left", on="Date")
    else:
        df["sent_mean"] = np.nan
        df["sent_count"] = 0

    # forward fill sentiment to ensure today's data has a value
    df["sent_mean"] = df["sent_mean"].fillna(method="ffill").fillna(0)
    df["sent_count"] = df["sent_count"].fillna(0)

    # lags
    for lag in range(1, lookback + 1):
        df[f"lag_close_{lag}"] = df["Close"].shift(lag)
        df[f"lag_return_{lag}"] = df["return_1d"].shift(lag)

    # target: next-day close
    df["target_next_close"] = df["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df

# ----------------------- UI -----------------------

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Ticker (yfinance)", value="AAPL")
    company_name = st.text_input("Company / Search query (optional)", value="Apple")
    start_date = st.date_input("Start date", value=(datetime.now() - timedelta(days=365)).date())
    end_date = st.date_input("End date", value=datetime.now().date())
with col2:
    news_api_key = st.text_input("NewsAPI.org Key (optional, improves results)")
    max_news = st.slider("Max news articles to fetch", 5, 50, 20)

if st.button("Run analysis"):
    with st.spinner("Fetching data and computing..."):
        # 1) fetch data
        price_df = fetch_price_data(ticker, start_date, end_date + timedelta(days=1))
        if price_df.empty:
            st.error("No price data found for that ticker & date range. Try another ticker or widen the date range.")
        else:
            news_df = get_news(company_name or ticker, api_key=news_api_key or None, max_articles=max_news)
            news_df = compute_sentiment(news_df)

            st.subheader("News fetched")
            if news_df.empty:
                st.warning("No news articles found.")
            else:
                st.dataframe(news_df[["publishedAt", "title", "source", "sentiment"]].sort_values("publishedAt", ascending=False))

            # 2) build features and model
            features = build_features(price_df, news_df, lookback=5)
            if features.empty:
                st.error("Not enough data to build features / train model.")
            else:
                X = features.drop(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "target_next_close"] , errors='ignore')
                y = features["target_next_close"]
                # simple train/test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                mse = mean_squared_error(y_test, preds)
                rmse = math.sqrt(mse)

                st.subheader("Model performance (holdout)")
                st.write(f"MAE: {mae:.4f}  —  RMSE: {rmse:.4f}")

                # Predict next day from latest row
                latest = features.iloc[-1:]
                X_latest = latest.drop(columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "target_next_close"], errors='ignore')
                next_price = model.predict(X_latest)[0]

                st.subheader("Next-day forecast")
                st.metric(label=f"Predicted next close for {ticker}", value=f"{next_price:.2f}")

                # plot historical actual vs predicted on test set
                test_plot_df = pd.DataFrame({"Date": X_test.index, "pred": preds, "actual": y_test.values})
                # but X_test.index are numeric; use corresponding dates from features
                test_plot_df["Date"] = features.iloc[X_test.index]["Date"].values
                fig = px.line(test_plot_df, x="Date", y=["actual", "pred"], labels={"value":"Price", "variable":"Series"}, title="Actual vs Predicted (holdout)")
                st.plotly_chart(fig, use_container_width=True)

                # show interactive price chart
                st.subheader("Price chart")
                fig2 = px.line(price_df, x="Date", y="Close", title=f"{ticker} Close Price")
                st.plotly_chart(fig2, use_container_width=True)

                # downloadable CSVs
                csv_buf = io.StringIO()
                news_df.to_csv(csv_buf, index=False)
                st.download_button("Download news CSV", csv_buf.getvalue(), file_name=f"{ticker}_news.csv")

                csv_buf2 = io.StringIO()
                price_df.to_csv(csv_buf2, index=False)
                st.download_button("Download price CSV", csv_buf2.getvalue(), file_name=f"{ticker}_prices.csv")

                st.success("Done ✅")

st.markdown("---")
st.write("**Notes & deployment**: This demo uses NewsAPI if you provide a key; otherwise it fetches Google News RSS as a fallback. The forecasting model is a simple RandomForest using price + aggregated sentiment features — treat it as a starting point, not financial advice.")
