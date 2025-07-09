import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import tempfile
import requests
import pandas as pd
import plotly.express as px
import gdown

st.set_page_config(
    page_title="🤖 AI Model Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_from_drive():
    try:
        file_id = "1WMmCh2bxuTiecevrQLTC2BeWJ9e-Gs_1"
        output_path = os.path.join(tempfile.gettempdir(), "model.h5")

        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

        model = tf.keras.models.load_model(output_path)
        return model, "model.h5"
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0).astype('float32') / 255.0
    return img_array

def create_prediction_chart(predictions, labels):
    df = pd.DataFrame({'Class': labels, 'Probability': predictions[0]})
    fig = px.bar(df, x='Class', y='Probability', color='Probability', color_continuous_scale='viridis')
    fig.update_layout(showlegend=False)
    return fig

def main():
    st.markdown('<h1 class="main-header">🤖 AI Model Predictor</h1>', unsafe_allow_html=True)

    with st.spinner("🔄 Downloading model from Google Drive..."):
        model, model_name = load_model_from_drive()

    if model is None:
        return

    st.success(f"✅ Model loaded: {model_name}")

    with st.sidebar:
        st.header("📊 Model Dashboard")
        st.info(f"📁 File: {model_name}")
        st.info(f"📐 Input: {model.input_shape}")
        st.info(f"📊 Output: {model.output_shape}")

        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
        show_probabilities = st.checkbox("Show All Probabilities", True)
        show_charts = st.checkbox("Show Prediction Charts", True)

        num_classes = model.output_shape[-1] if len(model.output_shape) > 1 else 1
        if num_classes > 1:
            default_labels = [f"Class {i+1}" for i in range(num_classes)]
            class_labels_input = st.text_area("Class labels (one per line):", "\n".join(default_labels), height=100)
            class_labels = [label.strip() for label in class_labels_input.split('\n') if label.strip()]
        else:
            class_labels = ["Output"]

    st.header("📸 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🚀 Predict", type="primary"):
            with st.spinner("🧠 AI is predicting..."):
                try:
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image, verbose=0)

                    if predictions.shape[-1] > 1:
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        predicted_class = class_labels[predicted_class_idx]

                        st.markdown(f"### 🏆 Prediction: `{predicted_class}`")
                        st.markdown(f"**Confidence**: {confidence:.2%}")

                        if confidence >= confidence_threshold:
                            st.success("✅ High confidence")
                        else:
                            st.warning("⚠️ Low confidence")

                        if show_probabilities:
                            for label, prob in zip(class_labels, predictions[0]):
                                st.write(f"{label}: {prob:.3f}")
                                st.progress(float(prob))

                        if show_charts:
                            fig = create_prediction_chart(predictions, class_labels)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.markdown(f"### Output: {predictions[0][0]:.4f}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
    else:
        st.info("👆 Upload an image to start predictions.")

if __name__ == "__main__":
    main()
