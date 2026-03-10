import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import time
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="InstruNet AI", page_icon="🎵", layout="wide")

# --------------------------------------------------
# PASTEL THEME
# --------------------------------------------------
st.markdown("""
<style>
.main {
background-color:#F4F9F9;
}
.title {
font-size:40px;
font-weight:700;
color:#1E2F2F;
}
.subtitle {
font-size:18px;
color:#5DA9A6;
}
.card {
background:#E6F4F1;
padding:20px;
border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎵 InstruNet AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Powered Musical Instrument Intelligence System</p>', unsafe_allow_html=True)
st.divider()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("instrunet_milestone2_final.keras")

with st.spinner("Loading AI model..."):
    model = load_model()
    time.sleep(1)

# --------------------------------------------------
# AUTO LABELS
# --------------------------------------------------
num_classes = model.output_shape[-1]
instrument_labels = [f"Instrument {i+1}" for i in range(num_classes)]

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.subheader("Upload Audio File 🎧")

uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav","mp3"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    with st.spinner("Analyzing audio..."):

        y, sr = librosa.load(uploaded_file, sr=22050)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_resized = np.resize(mel_spec_db,(128,128))
        mel_resized = mel_resized.reshape(1,128,128,1)

        prediction = model.predict(mel_resized)

        predicted_index = int(np.argmax(prediction))
        confidence_scores = prediction[0] * 100

        predicted_label = instrument_labels[predicted_index]
        confidence_value = confidence_scores[predicted_index]

        time.sleep(1)

# --------------------------------------------------
# RESULT
# --------------------------------------------------
    st.markdown(f"""
    <div class="card">
    <h2>🎼 Predicted Instrument: {predicted_label}</h2>
    <h3>Confidence: {confidence_value:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

# --------------------------------------------------
# WAVEFORM
# --------------------------------------------------
    st.subheader("Audio Waveform 🌊")

    waveform = px.line(y=y[:5000])
    st.plotly_chart(waveform,use_container_width=True)

# --------------------------------------------------
# MEL SPECTROGRAM
# --------------------------------------------------
    st.subheader("Mel Spectrogram 🔥")

    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_spec_db,sr=sr,x_axis='time',y_axis='mel')
    plt.colorbar(img,ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# SEGMENT ANALYSIS
# --------------------------------------------------
    st.subheader("Timeline Segment Analysis ⏱")

    duration=len(y)
    segment_length=duration//3

    segment_conf=[]

    for i in range(3):

        seg=y[i*segment_length:(i+1)*segment_length]

        mel_seg=librosa.feature.melspectrogram(y=seg,sr=sr)
        mel_seg_db=librosa.power_to_db(mel_seg,ref=np.max)

        mel_seg_resized=np.resize(mel_seg_db,(128,128))
        mel_seg_resized=mel_seg_resized.reshape(1,128,128,1)

        seg_pred=model.predict(mel_seg_resized)

        seg_conf=seg_pred[0][predicted_index]*100
        segment_conf.append(seg_conf)

    timeline=pd.DataFrame({
    "Segment":["Beginning","Middle","End"],
    "Confidence":segment_conf
    })

    timeline_chart=px.line(timeline,x="Segment",y="Confidence",markers=True)
    st.plotly_chart(timeline_chart,use_container_width=True)

# --------------------------------------------------
# INSTRUMENT HEALTH
# --------------------------------------------------
    st.subheader("Instrument Health Analysis 🩺")

    spectral_centroid=np.mean(librosa.feature.spectral_centroid(y=y,sr=sr))
    zero_crossing=np.mean(librosa.feature.zero_crossing_rate(y))
    rms=np.mean(librosa.feature.rms(y=y))

    health_score=(spectral_centroid*0.0001+rms*10-zero_crossing*10)

    if health_score>0.8:
        health="Excellent Resonance 🎼"
    elif health_score>0.5:
        health="Stable Harmonics 🎵"
    else:
        health="Needs Tuning ⚠"

    st.success(f"Instrument Health: {health}")

# --------------------------------------------------
# PROBABILITY ANALYSIS
# --------------------------------------------------
    st.subheader("Prediction Composition Analysis 📊")

    df=pd.DataFrame({
    "Instrument":instrument_labels,
    "Confidence":confidence_scores
    })

    bar=px.bar(df,x="Instrument",y="Confidence")
    st.plotly_chart(bar,use_container_width=True)

    pie=px.pie(df,names="Instrument",values="Confidence")
    st.plotly_chart(pie,use_container_width=True)

# --------------------------------------------------
# REPORT EXPORT
# --------------------------------------------------
    st.subheader("Download Reports 📄")

    report_data={
"Predicted Instrument": str(predicted_label),
"Confidence": float(confidence_value),
"Segment Confidence": [float(x) for x in segment_conf]
}

    json_data=json.dumps(report_data,indent=4)

    st.download_button(
    "Download JSON",
    json_data,
    file_name="analysis.json"
    )

    csv=df.to_csv(index=False)

    st.download_button(
    "Download CSV",
    csv,
    file_name="analysis.csv"
    )

# PDF

    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer)

    styles=getSampleStyleSheet()

    story=[]

    story.append(Paragraph("InstruNet AI Report",styles['Title']))
    story.append(Spacer(1,20))
    story.append(Paragraph(f"Predicted Instrument: {predicted_label}",styles['Normal']))
    story.append(Paragraph(f"Confidence: {confidence_value:.2f}%",styles['Normal']))
    story.append(Paragraph(f"Health Status: {health}",styles['Normal']))

    doc.build(story)

    st.download_button(
    "Download PDF",
    buffer.getvalue(),
    file_name="analysis_report.pdf"
    )