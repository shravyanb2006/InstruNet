import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("instrunet_milestone2_final.keras")

st.title("🎵 InstruNet AI - Music Instrument Recognition")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Load audio
    y, sr = librosa.load(uploaded_file, sr=22050)

    # Generate Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(img, ax=ax)
    st.pyplot(fig)

    # Resize for model
    mel_spec_db = np.resize(mel_spec_db, (128, 128))
    mel_spec_db = mel_spec_db.reshape(1, 128, 128, 1)

    prediction = model.predict(mel_spec_db)
    predicted_class = np.argmax(prediction)

    st.write("Predicted Instrument Class:", predicted_class)