import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import tempfile
import zipfile
import os
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="🤖 AI Model Predictor", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model_from_drive():
    try:
        file_id = "1WMmCh2bxuTiecevrQLTC2BeWJ9e-Gs_1"
        gdrive_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(gdrive_url)
        if response.status_code != 200:
            st.error(f"Failed to download model: HTTP {response.status_code}")
            return None, None
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "model.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            for root, dirs, files in os.walk(tmp_dir):
                for file in files:
                    if file.endswith('.h5'):
                        model_path = os.path.join(root, file)
                        model = tf.keras.models.load_model(model_path)
                        return model, file
        st.error("No .h5 model file found in the zip archive")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(160, 240)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, -1).astype('float32')  # Flattened
    return img_array

def create_prediction_chart(predictions, labels):
    df = pd.DataFrame({'Class': labels, 'Probability': predictions[0]})
    fig = px.bar(df, x='Class', y='Probability', color='Probability',
                 color_continuous_scale='viridis',
                 title='Prediction Probabilities')
    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig

def main():
    st.title("🤖 AI Model Predictor")

    with st.spinner("Loading model..."):
        model, model_name = load_model_from_drive()

    if model is None:
        st.stop()

    st.sidebar.header("📊 Model Dashboard")
    st.sidebar.write(f"**File**: {model_name}")
    st.sidebar.write(f"**Input Shape**: {model.input_shape}")
    st.sidebar.write(f"**Output Shape**: {model.output_shape}")

    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    show_probs = st.sidebar.checkbox("Show All Probabilities", True)
    show_chart = st.sidebar.checkbox("Show Prediction Charts", True)

    num_classes = model.output_shape[-1] if len(model.output_shape) > 1 else 1
    if num_classes > 1:
        default_labels = [f"Class {i+1}" for i in range(num_classes)]
        class_labels_input = st.sidebar.text_area("Enter class labels:", "\n".join(default_labels))
        class_labels = [label.strip() for label in class_labels_input.split('\n') if label.strip()]
    else:
        class_labels = ["Output"]

    st.header("📸 Upload Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.subheader("🎯 AI Prediction")
        if st.button("🚀 Predict"):
            with st.spinner("Predicting..."):
                try:
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image, verbose=0)

                    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence = np.max(predictions[0])
                        predicted_class = class_labels[predicted_class_idx]
                        st.write(f"**Predicted Class:** {predicted_class}")
                        st.write(f"**Confidence:** {confidence:.2%}")
                        if confidence >= confidence_threshold:
                            st.success("High Confidence")
                        else:
                            st.warning("Low Confidence")

                        if show_probs:
                            st.subheader("📊 Class Probabilities")
                            for i, (label, prob) in enumerate(zip(class_labels, predictions[0])):
                                st.write(f"{label}: {prob:.3f}")
                                st.progress(float(prob))

                        if show_chart:
                            st.subheader("📈 Prediction Chart")
                            fig = create_prediction_chart(predictions, class_labels)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write(f"**Prediction:** {predictions[0][0]:.4f}")

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.info("Try adjusting the image or verifying model input shape.")

if __name__ == "__main__":
    main()
