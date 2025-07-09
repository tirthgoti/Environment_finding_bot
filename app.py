import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# ------------------ CONFIG ------------------
st.set_page_config(page_title="🌍 Land Cover Classifier", layout="wide")

MODEL_DRIVE_ID = "1WMmCh2bxuTiecevrQLTC2BeWJ9e-Gs_1"
MODEL_FILE = "Modelenv.v1.h5"

# ------------------ DOWNLOAD MODEL FROM DRIVE ------------------
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_FILE):
        st.info("⬇️ Downloading model from Google Drive...")
        gdown.download(id=MODEL_DRIVE_ID, output=MODEL_FILE, quiet=False)
    model = tf.keras.models.load_model(MODEL_FILE)
    return model

model = download_model()

# ------------------ PREPROCESS IMAGE ------------------
def preprocess(image, target_size=(224, 224)):
    image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# ------------------ LABELS (customize if needed) ------------------
class_labels = ["Urban", "Forest", "Water", "Agriculture", "Barren", "Others"]

# ------------------ STREAMLIT UI ------------------
st.title("🌍 Environmental Monitoring & Land Cover Classification")
st.markdown("Upload a satellite image and get predicted land cover type.")

uploaded_file = st.file_uploader("📤 Upload a satellite image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    if st.button("🚀 Predict"):
        with st.spinner("Analyzing..."):
            processed = preprocess(image)
            preds = model.predict(processed)[0]
            predicted_idx = np.argmax(preds)
            predicted_label = class_labels[predicted_idx]
            confidence = preds[predicted_idx]

            st.success(f"✅ **Prediction:** {predicted_label} ({confidence:.2%} confidence)")

            st.subheader("📊 Class Probabilities")
            for label, prob in zip(class_labels, preds):
                st.write(f"- **{label}**: {prob:.2%}")
