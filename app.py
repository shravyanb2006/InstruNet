import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib import colors
import io
import time

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="InstruNet AI", page_icon="🎵", layout="wide")

# -------------------------------------------------------
# THEME STYLING
# -------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #F4F9F9;
}
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
# LABELS (EDIT ACCORDING TO YOUR TRAINING ORDER)
# -------------------------------------------------------
instrument_labels = [
    "Piano", "Guitar", "Drums", "Violin",
    "Flute", "Saxophone", "Trumpet", "Cello"
]

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
        predicted_index = np.argmax(prediction)
        confidence_scores = prediction[0] * 100
        predicted_label = instrument_labels[predicted_index]

        time.sleep(1)

    # -------------------------------------------------------
    # RESULT CARD
    # -------------------------------------------------------
    st.markdown(f"""
    <div class="card">
        <h2>🎼 Predicted Instrument: {predicted_label}</h2>
        <h4>Confidence: {confidence_scores[predicted_index]:.2f}%</h4>
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