import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import helper functions implemented in rnn_lstm_app.py
from rnn_lstm_app import (
    load_csv,
    prepare_time_series,
    create_lstm,
    prepare_texts,
    create_rnn,
)


st.set_page_config(page_title="RNN & LSTM Demo", layout="wide")
st.title("RNN & LSTM Interactive Demo")

BASE = Path(__file__).parent
sample_stock = BASE / "sample_stock_prices.csv"
sample_sentiment = BASE / "sample_sentiment_text.csv"


def run_lstm_ui(df: pd.DataFrame):
    st.subheader("LSTM Time Series (Close prices)")
    st.write("Data preview:")
    st.dataframe(df.head())

    seq_len = st.sidebar.slider("Sequence length", min_value=3, max_value=50, value=10)
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=5)

    if st.button("Train LSTM"):
        try:
            X, y, scaler = prepare_time_series(df['Close'].values, seq_len=seq_len)
        except ValueError as ve:
            st.error(str(ve))
            return

        model = create_lstm(seq_len)
        with st.spinner("Training LSTM (small demo)..."):
            model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)

        preds = model.predict(X)
        preds_inv = scaler.inverse_transform(preds)

        results = pd.DataFrame({
            'Actual': df['Close'].values[seq_len: seq_len + len(preds_inv)],
            'Predicted': preds_inv.flatten()
        })
        st.write("Results (first 50 rows):")
        st.dataframe(results.head(50))
        st.line_chart(results)

        # Predict next value (use scaler from training to build last input)
        if len(df['Close'].values) < seq_len:
            st.warning("Not enough data to predict next value (increase dataset or reduce sequence length).")
        else:
            last_values = df['Close'].values[-seq_len:]
            # scale using the same scaler used for training
            scaled_last = scaler.transform(np.array(last_values).reshape(-1, 1))
            x_input = scaled_last.reshape(1, seq_len, 1)
            next_pred = model.predict(x_input)
            next_price = scaler.inverse_transform(next_pred)[0][0]
            st.success(f"Predicted next closing value (demo): {next_price:.4f}")


def run_rnn_ui(df: pd.DataFrame):
    st.subheader("RNN Sentiment (SimpleRNN)")
    st.write("Data preview:")
    st.dataframe(df.head())

    epochs = st.sidebar.slider("Epochs (RNN)", min_value=1, max_value=50, value=5)

    if st.button("Train RNN"):
        texts = df['Text'].astype(str).values
        if 'Sentiment' in df.columns:
            if df['Sentiment'].dtype == object:
                labels = (df['Sentiment'].str.lower() == 'positive').astype(int).values
            else:
                labels = df['Sentiment'].astype(int).values
        else:
            st.error("No 'Sentiment' column found in CSV. Provide binary labels (0/1) or 'positive'/'negative'.")
            return

        X, tokenizer = prepare_texts(texts)
        vocab_size = min(1000, len(tokenizer.word_index) + 1)
        model = create_rnn(vocab_size=vocab_size)
        with st.spinner("Training RNN (small demo)..."):
            model.fit(X, labels, epochs=epochs, batch_size=8, verbose=0)

        preds = (model.predict(X) > 0.5).astype(int).flatten()
        df_out = df.copy()
        df_out['Predicted'] = preds
        st.write("Predictions:")
        st.dataframe(df_out)

        acc = (preds == labels).mean()
        st.success(f"Training accuracy (demo): {acc*100:.2f}%")


def main():
    st.sidebar.header("Dataset options")
    use_sample = st.sidebar.checkbox("Use sample data when no upload provided", value=True)

    app_mode = st.sidebar.selectbox("Choose demo", ["LSTM Time Series", "RNN Sentiment"])

    if app_mode == "LSTM Time Series":
        uploaded = st.file_uploader("Upload CSV with 'Close' column", type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            run_lstm_ui(df)
        elif use_sample and sample_stock.exists():
            df = pd.read_csv(sample_stock)
            run_lstm_ui(df)
        else:
            st.info("Upload a CSV with a 'Close' column or place sample_stock_prices.csv in the folder.")

    else:
        uploaded = st.file_uploader("Upload CSV with 'Text' and 'Sentiment' columns", type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            run_rnn_ui(df)
        elif use_sample and sample_sentiment.exists():
            df = pd.read_csv(sample_sentiment)
            run_rnn_ui(df)
        else:
            st.info("Upload a CSV with 'Text' and 'Sentiment' columns or place sample_sentiment_text.csv in the folder.")


if __name__ == '__main__':
    main()
