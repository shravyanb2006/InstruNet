import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import json
import io
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="InstruNet AI", page_icon="🎵", layout="wide")

# -----------------------------
# UI STYLE
# -----------------------------

st.markdown("""
<style>
.main{
background-color:#F4FBFB;
}

h1{
color:#2C6E6E;
}

.card{
background:#E8F6F6;
padding:20px;
border-radius:15px;
margin-bottom:20px;
box-shadow:0px 4px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🎵 InstruNet AI")
st.caption("AI Powered Musical Instrument Classification & Health Analysis")

st.divider()

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("instrunet_milestone2_final.keras")

model = load_model()

num_classes = model.output_shape[-1]
instrument_labels = [f"Instrument {i}" for i in range(num_classes)]

# -----------------------------
# AUDIO UPLOAD
# -----------------------------

st.subheader("Upload Instrument Audio")

uploaded_file = st.file_uploader("Upload WAV or MP3", type=["wav","mp3"])

if uploaded_file:

    st.audio(uploaded_file)

    with st.spinner("AI analyzing audio..."):
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
        confidence = float(probabilities[predicted_index])

# -----------------------------
# RESULT
# -----------------------------

    st.markdown(f"""
    <div class="card">
    <h2>🎼 Predicted Instrument: {predicted_label}</h2>
    <h3>Confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# WAVEFORM
# -----------------------------

    st.subheader("Audio Waveform")

    fig_wave = plt.figure()
    plt.plot(y[:8000])
    plt.title("Audio Waveform")
    st.pyplot(fig_wave)

# -----------------------------
# MEL SPECTROGRAM
# -----------------------------

    st.subheader("Mel Spectrogram")

    fig_spec, ax = plt.subplots()

    img = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(img, ax=ax)

    st.pyplot(fig_spec)

# -----------------------------
# SEGMENT ANALYSIS
# -----------------------------

    st.subheader("3-Segment Timeline Analysis")

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

    fig_time = plt.figure()

    plt.plot(["Beginning","Middle","End"], seg_conf, marker='o')
    plt.ylabel("Confidence")
    plt.title("Segment Prediction Confidence")

    st.pyplot(fig_time)

# -----------------------------
# INSTRUMENT HEALTH
# -----------------------------

    st.subheader("Instrument Health")

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y,sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    health_score = spectral_centroid*0.0001 + rms*10 - zcr*10

    if health_score > 0.8:
        health = "Excellent Resonance"
    elif health_score > 0.5:
        health = "Stable Harmonics"
    else:
        health = "Needs Tuning"

    st.success(f"Health Status: {health}")

# -----------------------------
# PROBABILITY DISTRIBUTION
# -----------------------------

    st.subheader("Prediction Probability Distribution")

    prob_df = pd.DataFrame({
        "Instrument":instrument_labels,
        "Confidence":probabilities
    })

    fig_bar = plt.figure()

    plt.bar(instrument_labels, probabilities)
    plt.xticks(rotation=45)
    plt.ylabel("Confidence %")
    plt.title("Prediction Confidence")

    st.pyplot(fig_bar)

    fig_pie = plt.figure()

    plt.pie(probabilities, labels=instrument_labels, autopct='%1.1f%%')
    plt.title("Prediction Composition")

    st.pyplot(fig_pie)

# -----------------------------
# AI EXPLANATION
# -----------------------------

    st.subheader("AI Explanation")

    explanation=f"""
The AI predicted **{predicted_label}** because the sound contained frequency 
patterns and harmonic signatures similar to those learned for this instrument.
The mel spectrogram reveals the energy distribution across frequency bands
which strongly matches the model's training features.
"""

    st.info(explanation)

# -----------------------------
# REPORT DATA
# -----------------------------

    report_data={
        "Instrument":predicted_label,
        "Confidence":confidence,
        "Segment Confidence":seg_conf,
        "Health":health
    }

# -----------------------------
# DOWNLOAD REPORTS
# -----------------------------

    st.subheader("Download Reports")

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

    wave_img="wave.png"
    spec_img="spec.png"
    bar_img="bar.png"
    pie_img="pie.png"
    time_img="timeline.png"

    fig_wave.savefig(wave_img)
    fig_spec.savefig(spec_img)
    fig_bar.savefig(bar_img)
    fig_pie.savefig(pie_img)
    fig_time.savefig(time_img)

    buffer=io.BytesIO()

    doc=SimpleDocTemplate(buffer)

    styles=getSampleStyleSheet()

    story=[]

    story.append(Paragraph("InstruNet AI Report",styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"Instrument: {predicted_label}",styles['Normal']))
    story.append(Paragraph(f"Confidence: {confidence:.2f}%",styles['Normal']))
    story.append(Paragraph(f"Health: {health}",styles['Normal']))

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