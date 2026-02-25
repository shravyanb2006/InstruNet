import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="InstruNet AI", page_icon="🎵", layout="wide")

# -------------------------------------------------------
# THEME
# -------------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #F4F9F9;
}
.big-title {
    font-size: 42px;
    font-weight: 700;
    color: #1E2F2F;
}
.subtitle {
    font-size: 18px;
    color: #5DA9A6;
}
.card {
    background-color: #E6F4F1;
    padding: 20px;
    border-radius: 15px;
    color: #1E2F2F;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown('<p class="big-title">🎵 InstruNet AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Musical Instrument Intelligence System</p>', unsafe_allow_html=True)
st.divider()

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("instrunet_milestone2_final.keras")

with st.spinner("🎼 Loading AI model..."):
    model = load_model()
    time.sleep(1)

# -------------------------------------------------------
# AUTO DETECT CLASS COUNT
# -------------------------------------------------------
num_classes = model.output_shape[-1]

# Generate default labels automatically
instrument_labels = [f"Instrument {i}" for i in range(num_classes)]

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
st.subheader("🎧 Upload Audio File")
uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    with st.spinner("🔍 Analyzing audio signal..."):
        y, sr = librosa.load(uploaded_file, sr=22050)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_resized = np.resize(mel_spec_db, (128, 128))
        mel_spec_resized = mel_spec_resized.reshape(1, 128, 128, 1)

        prediction = model.predict(mel_spec_resized)
        predicted_index = int(np.argmax(prediction))
        confidence_scores = prediction[0] * 100

        predicted_label = instrument_labels[predicted_index]
        confidence_value = confidence_scores[predicted_index]

        time.sleep(1)

    # -------------------------------------------------------
    # RESULT CARD
    # -------------------------------------------------------
    st.markdown(f"""
    <div class="card">
        <h2>🎼 Predicted Instrument: {predicted_label}</h2>
        <h4>Confidence: {confidence_value:.2f}%</h4>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # -------------------------------------------------------
    # WAVEFORM
    # -------------------------------------------------------
    st.subheader("🌊 Audio Waveform")
    fig_wave = px.line(y=y[:5000], title="Waveform Snapshot")
    st.plotly_chart(fig_wave, use_container_width=True)

    # -------------------------------------------------------
    # MEL SPECTROGRAM
    # -------------------------------------------------------
    st.subheader("🔥 Mel Spectrogram")
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(img, ax=ax)
    st.pyplot(fig)

    # -------------------------------------------------------
    # CONFIDENCE DISTRIBUTION
    # -------------------------------------------------------
    st.subheader("📊 Prediction Confidence Distribution")

    df = pd.DataFrame({
        "Instrument": instrument_labels,
        "Confidence (%)": confidence_scores
    })

    fig_bar = px.bar(df, x="Instrument", y="Confidence (%)",
                     color="Confidence (%)",
                     color_continuous_scale="Teal")

    st.plotly_chart(fig_bar, use_container_width=True)