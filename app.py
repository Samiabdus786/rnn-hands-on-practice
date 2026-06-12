import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="RNN – Time Series", page_icon="📈", layout="wide")

st.title("📈 RNN – Recurrent Neural Network for Time Series Prediction")
st.markdown("A **SimpleRNN** learns to predict the next value in a noisy sine wave — a classic sequence learning demo.")

st.sidebar.header("⚙️ Settings")
seq_len    = st.sidebar.slider("Sequence Length (look-back)", 10, 50, 20)
n_samples  = st.sidebar.slider("Total Samples",  500, 5000, 1000, step=500)
rnn_units  = st.sidebar.slider("RNN Units",       16,  128,  32,  step=16)
epochs     = st.sidebar.slider("Epochs",           5,   50,  20)
noise_std  = st.sidebar.slider("Noise Std",        0.0,  0.5, 0.1, step=0.05)

# ── Generate data ──────────────────────────────────────────────────────────
@st.cache_data
def make_data(n, noise, seq):
    t       = np.linspace(0, 4 * np.pi, n)
    signal  = np.sin(t) + np.random.normal(0, noise, n)
    X, y    = [], []
    for i in range(len(signal) - seq):
        X.append(signal[i:i+seq])
        y.append(signal[i+seq])
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], signal

st.subheader("🔢 Generated Sine Wave Data")
X_train, X_test, y_train, y_test, full_signal = make_data(n_samples, noise_std, seq_len)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(full_signal[:200], label="Signal", lw=1.5)
ax.set_title("First 200 points of the training signal")
ax.legend()
st.pyplot(fig)

st.write(f"Train size: **{len(X_train)}** | Test size: **{len(X_test)}**")

if st.button("🚀 Train RNN"):
    with st.spinner("Training …"):
        model = keras.Sequential([
            layers.SimpleRNN(rnn_units, activation="tanh", return_sequences=True,
                             input_shape=(seq_len, 1)),
            layers.SimpleRNN(rnn_units // 2, activation="tanh"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X_train, y_train, epochs=epochs,
                            batch_size=32, validation_split=0.1, verbose=0)

    mse  = model.evaluate(X_test, y_test, verbose=0)
    preds = model.predict(X_test, verbose=0).flatten()
    st.success(f"✅ Test MSE: **{mse:.5f}** | RMSE: **{np.sqrt(mse):.5f}**")

    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history["loss"],     label="Train Loss")
        ax2.plot(history.history["val_loss"], label="Val Loss")
        ax2.set_title("Training Loss (MSE)"); ax2.legend()
        st.pyplot(fig2)
    with col2:
        fig3, ax3 = plt.subplots()
        ax3.plot(y_test[:100],  label="Actual",    lw=1.5)
        ax3.plot(preds[:100],   label="Predicted", lw=1.5, linestyle="--")
        ax3.set_title("Actual vs Predicted (first 100 test points)"); ax3.legend()
        st.pyplot(fig3)

st.markdown("---")
st.markdown("**Architecture:** SimpleRNN → SimpleRNN → Dense(1) | Loss: MSE | Optimizer: Adam")
