import streamlit as st
import numpy as np
import os
import gdown
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# === CONFIG ===
MODEL_ID = "1WMmCh2bxuTiecevrQLTC2BeWJ9e-Gs_1"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "satellite_model_v1.h5"
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']
IMAGE_SIZE = (256, 256)

# === PAGE SETUP ===
st.set_page_config(page_title="🌍 Satellite Image Classifier", layout="centered")
st.title("🌍 Satellite Image Classifier")
st.markdown("Upload a satellite image to classify it as **Cloudy**, **Desert**, **Green Area**, or **Water**.")

# === MODEL LOADING ===
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("🔄 Downloading model... (one-time)")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return None
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

model = download_and_load_model()
if model is None:
    st.stop()

# === IMAGE UPLOAD ===
uploaded_file = st.file_uploader("📤 Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # === PREDICT BUTTON ===
    if st.button("🔍 Classify Image"):
        with st.spinner("Analyzing image..."):
            try:
                # Preprocess
                img_array = img_to_array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = model.predict(img_array)[0]
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = np.max(prediction)

                # === DISPLAY RESULTS ===
                st.success(f"✅ **Prediction:** {predicted_class}")
                st.metric("🔒 Confidence", f"{confidence * 100:.2f}%")

                # Chart
                st.markdown("### 📊 Prediction Probabilities")
                prob_df = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability": prediction
                }).sort_values("Probability", ascending=False)

                st.bar_chart(prob_df.set_index("Class"))

                # Table
                prob_df["Confidence %"] = (prob_df["Probability"] * 100).round(2).astype(str) + "%"
                st.dataframe(prob_df[["Class", "Confidence %"]].reset_index(drop=True))

            except Exception as e:
                st.error(f"Prediction error: {e}")
