import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import json
import io
import time
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide"
)

# -------------------------------------------------
# UI STYLE
# -------------------------------------------------

st.markdown("""
<style>

.main {
background-color:#F5FBFB;
}

.title{
font-size:42px;
font-weight:700;
color:#233;
}

.subtitle{
font-size:18px;
color:#5DA9A6;
}

.card{
background:#E7F4F3;
padding:25px;
border-radius:15px;
margin-bottom:20px;
box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎵 InstruNet AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI Powered Musical Instrument Intelligence Platform</p>', unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("instrunet_milestone2_final.keras")

with st.spinner("Loading AI model..."):
    model = load_model()

num_classes = model.output_shape[-1]
instrument_labels = [f"Instrument {i}" for i in range(num_classes)]

# -------------------------------------------------
# AUDIO UPLOAD
# -------------------------------------------------

st.subheader("Upload Audio 🎧")

uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav","mp3"])

if uploaded_file:

    st.audio(uploaded_file)

    with st.spinner("Analyzing audio with AI..."):
        time.sleep(1)

        y, sr = librosa.load(uploaded_file, sr=22050)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_resize = np.resize(mel_db,(128,128))
        mel_resize = mel_resize.reshape(1,128,128,1)

        prediction = model.predict(mel_resize)

        predicted_index = int(np.argmax(prediction))
        probabilities = prediction[0]*100

        predicted_label = instrument_labels[predicted_index]
        confidence = probabilities[predicted_index]

# -------------------------------------------------
# RESULT CARD
# -------------------------------------------------

    st.markdown(f"""
    <div class="card">
    <h2>🎼 Predicted Instrument: {predicted_label}</h2>
    <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# WAVEFORM
# -------------------------------------------------

    st.subheader("Audio Waveform 🌊")

    waveform_df = pd.DataFrame({
        "Amplitude":y[:8000]
    })

    waveform_chart = px.line(
        waveform_df,
        y="Amplitude",
        title="Audio Waveform"
    )

    st.plotly_chart(waveform_chart, use_container_width=True)

# -------------------------------------------------
# MEL SPECTROGRAM
# -------------------------------------------------

    st.subheader("Mel Spectrogram 🔥")

    fig, ax = plt.subplots()

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(img, ax=ax)

    st.pyplot(fig)

# -------------------------------------------------
# SEGMENT ANALYSIS
# -------------------------------------------------

    st.subheader("Timeline Segment Analysis ⏱")

    length = len(y)
    segment = length//3

    seg_conf=[]

    for i in range(3):

        seg = y[i*segment:(i+1)*segment]

        mel_seg = librosa.feature.melspectrogram(y=seg,sr=sr)
        mel_seg_db = librosa.power_to_db(mel_seg,ref=np.max)

        mel_seg_resize = np.resize(mel_seg_db,(128,128))
        mel_seg_resize = mel_seg_resize.reshape(1,128,128,1)

        seg_pred = model.predict(mel_seg_resize)

        seg_conf.append(float(seg_pred[0][predicted_index]*100))

    timeline_df = pd.DataFrame({
        "Segment":["Beginning","Middle","End"],
        "Confidence":seg_conf
    })

    timeline_chart = px.line(
        timeline_df,
        x="Segment",
        y="Confidence",
        markers=True
    )

    st.plotly_chart(timeline_chart, use_container_width=True)

# -------------------------------------------------
# INSTRUMENT HEALTH
# -------------------------------------------------

    st.subheader("Instrument Health 🩺")

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y,sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    health_score = spectral_centroid*0.0001 + rms*10 - zcr*10

    if health_score > 0.8:
        health = "Excellent Resonance 🎼"
    elif health_score > 0.5:
        health = "Stable Harmonics 🎵"
    else:
        health = "Needs Tuning ⚠"

    st.success(f"Health Status: {health}")

# -------------------------------------------------
# PROBABILITY DISTRIBUTION
# -------------------------------------------------

    st.subheader("Prediction Composition 📊")

    prob_df = pd.DataFrame({
        "Instrument":instrument_labels,
        "Confidence":probabilities
    })

    bar_chart = px.bar(
        prob_df,
        x="Instrument",
        y="Confidence",
        color="Confidence"
    )

    st.plotly_chart(bar_chart,use_container_width=True)

    pie_chart = px.pie(
        prob_df,
        names="Instrument",
        values="Confidence"
    )

    st.plotly_chart(pie_chart,use_container_width=True)

# -------------------------------------------------
# AI EXPLANATION
# -------------------------------------------------

    st.subheader("AI Explanation 🧠")

    explanation = f"""
The AI predicted **{predicted_label}** because the audio contained 
spectral patterns and harmonic structures that closely match the learned 
features of this instrument. The mel spectrogram shows energy distribution 
across frequencies that aligns with this instrument's acoustic signature.
"""

    st.info(explanation)

# -------------------------------------------------
# SAVE IMAGES FOR PDF
# -------------------------------------------------

    wave_img="wave.png"
    spec_img="spec.png"
    bar_img="bar.png"
    pie_img="pie.png"
    time_img="timeline.png"

    waveform_chart.write_image(wave_img)
    bar_chart.write_image(bar_img)
    pie_chart.write_image(pie_img)
    timeline_chart.write_image(time_img)

    fig.savefig(spec_img)

# -------------------------------------------------
# REPORT DOWNLOAD
# -------------------------------------------------

    st.subheader("Download Reports 📄")

    report_data={
        "Instrument":predicted_label,
        "Confidence":float(confidence),
        "Segment Confidence":seg_conf,
        "Health":health
    }

# JSON

    json_data=json.dumps(report_data,indent=4)

    st.download_button(
        "Download JSON",
        json_data,
        file_name="analysis.json"
    )

# CSV

    csv_data=prob_df.to_csv(index=False)

    st.download_button(
        "Download CSV",
        csv_data,
        file_name="analysis.csv"
    )

# PDF

    buffer=io.BytesIO()

    doc=SimpleDocTemplate(buffer)

    styles=getSampleStyleSheet()

    story=[]

    story.append(Paragraph("InstruNet AI Analysis Report",styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"Predicted Instrument: {predicted_label}",styles['Normal']))
    story.append(Paragraph(f"Confidence: {confidence:.2f}%",styles['Normal']))
    story.append(Paragraph(f"Health Status: {health}",styles['Normal']))

    story.append(Spacer(1,20))

    story.append(Image(wave_img,width=400,height=200))
    story.append(Image(spec_img,width=400,height=200))
    story.append(Image(time_img,width=400,height=200))
    story.append(Image(bar_img,width=400,height=200))
    story.append(Image(pie_img,width=400,height=200))

    doc.build(story)

    st.download_button(
        "Download PDF",
        buffer.getvalue(),
        file_name="analysis_report.pdf"
    )