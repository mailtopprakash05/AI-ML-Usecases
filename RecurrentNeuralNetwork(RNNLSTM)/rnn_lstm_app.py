import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_csv(path):
    """Utility to load CSV into a DataFrame."""
    return pd.read_csv(path)


def prepare_time_series(data_series, seq_len=10):
    """Prepare time-series data for LSTM.

    Args:
        data_series: 1D numpy array or pd.Series of numeric values.
        seq_len: number of timesteps for each sample.

    Returns:
        X: numpy array shape (n_samples, seq_len, 1)
        y: numpy array shape (n_samples, 1)
        scaler: fitted MinMaxScaler
    """
    arr = np.array(data_series).astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr.reshape(-1, 1))

    n_samples = len(scaled) - seq_len
    if n_samples <= 0:
        raise ValueError(f"Not enough data points ({len(scaled)}) for sequence length {seq_len}.")

    X, y = [], []
    for i in range(n_samples):
        X.append(scaled[i : i + seq_len])
        y.append(scaled[i + seq_len])

    X = np.array(X)
    y = np.array(y)

    # Ensure X has shape (n_samples, seq_len, 1)
    if X.ndim == 2:
        # shape might be (n_samples, seq_len) => add channel dim
        X = X.reshape(X.shape[0], X.shape[1], 1)
    elif X.ndim == 3 and X.shape[-1] != 1:
        # If last dim isn't 1, try to reshape accordingly
        X = X.reshape(X.shape[0], X.shape[1], 1)

    # Ensure y is column vector
    y = y.reshape(-1, 1)

    return X, y, scaler


def create_lstm(seq_len, features=1, units=50):
    model = Sequential()
    model.add(LSTM(units, input_shape=(seq_len, features)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def prepare_texts(texts, max_words=1000, max_len=20):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seq, maxlen=max_len)
    return X, tokenizer


def create_rnn(vocab_size=1000, max_len=20, embed_dim=16, rnn_units=32):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len))
    model.add(SimpleRNN(rnn_units))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def demo_time_series_workflow(csv_path=None):
    """Demo: load stock CSV (expects a 'Close' column), train small LSTM and predict."""
    if csv_path is None:
        # generate synthetic sine data
        data = np.sin(np.linspace(0, 50, 500))
        df = pd.DataFrame({"Close": data})
    else:
        df = load_csv(csv_path)

    seq_len = 10
    X, y, scaler = prepare_time_series(df['Close'].values, seq_len=seq_len)
    model = create_lstm(seq_len)
    model.fit(X, y, epochs=5, batch_size=16, verbose=1)

    # Predict on the training set for demonstration
    preds = model.predict(X)
    preds_inv = scaler.inverse_transform(preds)
    return df, preds_inv


def demo_text_workflow(csv_path=None):
    """Demo: load sentiment CSV (expects 'Text' and 'Sentiment' columns), train RNN."""
    if csv_path is None:
        texts = [
            "I love this product",
            "This is the worst",
            "Amazing experience",
            "Not good at all",
        ]
        sentiments = np.array([1, 0, 1, 0])
        df = pd.DataFrame({"Text": texts, "Sentiment": sentiments})
    else:
        df = load_csv(csv_path)
        # ensure sentiments are 0/1
        if df['Sentiment'].dtype == object:
            df['Sentiment'] = (df['Sentiment'].str.lower() == 'positive').astype(int)
        sentiments = df['Sentiment'].values

    X, tokenizer = prepare_texts(df['Text'].astype(str).values)
    model = create_rnn(vocab_size=min(1000, len(tokenizer.word_index) + 1))
    model.fit(X, sentiments, epochs=5, batch_size=4, verbose=1)
    preds = (model.predict(X) > 0.5).astype(int).flatten()
    df['Predicted'] = preds
    return df


if __name__ == '__main__':
    # Quick smoke demos
    print("Running LSTM demo (time series)...")
    df_ts, preds = demo_time_series_workflow()
    print("Time series demo done. Predictions shape:", preds.shape)

    print("\nRunning RNN demo (text/sentiment)...")
    df_txt = demo_text_workflow()
    print(df_txt)

