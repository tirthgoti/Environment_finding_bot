import streamlit as st
import numpy as np
import pandas as pd
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# === CONFIG ===
MODEL_ID = "1WMmCh2bxuTiecevrQLTC2BeWJ9e-Gs_1"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "satellite_model.h5"
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']
IMAGE_SIZE = (256, 256)

# === PAGE ===
st.set_page_config(page_title="🛰️ Satellite Classifier", layout="wide")
st.markdown("<h1 style='text-align: center;'>🛰️ Satellite Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to classify it as Cloudy, Desert, Green Area, or Water.</p>", unsafe_allow_html=True)

# === MODEL LOADING ===
@st.cache_resource
def load_satellite_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_satellite_model()

# === LAYOUT ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Satellite Image")
    uploaded_file = st.file_uploader("Choose a satellite image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Classify Image"):
            with st.spinner("Running model prediction..."):
                # Preprocess
                img_array = img_to_array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                prediction = model.predict(img_array)[0]
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = float(np.max(prediction))

                # Store in session
                st.session_state.result = {
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "all_probs": prediction
                }

with col2:
    st.subheader("🔬 Prediction Results")

    if "result" in st.session_state:
        result = st.session_state.result

        st.success(f"🧠 **Predicted Class:** {result['prediction']}")
        st.metric("📈 Confidence", f"{result['confidence'] * 100:.2f}%")
        st.progress(result['confidence'])

        st.markdown("### 📊 Class Probabilities")
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": result['all_probs']
        }).sort_values(by="Probability", ascending=False)

        st.bar_chart(df.set_index("Class"))

        df["Confidence (%)"] = (df["Probability"] * 100).round(2)
        st.dataframe(df[["Class", "Confidence (%)"]], use_container_width=True)
    else:
        st.info("⬆️ Upload an image and click 'Classify Image' to view prediction.")

# === FOOTER ===
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9rem;'>Built with ❤️ using Streamlit & TensorFlow</p>",
    unsafe_allow_html=True
)
