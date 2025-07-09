import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import zipfile
import os
import tempfile
import requests
from io import BytesIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .confidence-high {
        color: #00FF7F;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-low {
        color: #FFB347;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_from_zip():
    """Load model from zip file in GitHub repository"""
    try:
        
        zip_url = "https://github.com/tirthgoti/Environment_finding_bot/raw/main/Environmental_Monitoring_and_Classification_of_Land_Cover_Using_Satellite_Images_.zip"
        
        
        response = requests.get(zip_url)
        
        if response.status_code == 200:
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                
                zip_path = os.path.join(tmp_dir, "model.zip")
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
            
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                
                for file in os.listdir(tmp_dir):
                    if file.endswith('.h5'):
                        model_path = os.path.join(tmp_dir, file)
                        model = tf.keras.models.load_model(model_path)
                        return model, file
                
                
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        if file.endswith('.h5'):
                            model_path = os.path.join(root, file)
                            model = tf.keras.models.load_model(model_path)
                            return model, file
                
                st.error("No .h5 model file found in the zip archive")
                return None, None
        else:
            st.error(f"Failed to download model: HTTP {response.status_code}")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    
    image = image.resize(target_size)
    
    
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def create_prediction_chart(predictions, labels):
    """Create interactive prediction chart"""
    df = pd.DataFrame({
        'Class': labels,
        'Probability': predictions[0]
    })
    
    fig = px.bar(
        df, 
        x='Class', 
        y='Probability',
        color='Probability',
        color_continuous_scale='viridis',
        title='Prediction Probabilities'
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(size=16, color='white')
    )
    
    return fig

def main():
    
    st.markdown('<h1 class="main-header">🤖 AI Model Predictor</h1>', unsafe_allow_html=True)
    
    
    with st.spinner('🔄 Loading model from GitHub...'):
        model, model_name = load_model_from_zip()
    
    if model is None:
        st.error("❌ Failed to load model. Please check your GitHub repository URL.")
        st.info("📝 Make sure to update the zip_url in the code with your actual GitHub raw file URL")
        return
    
    
    st.success(f"✅ Model loaded successfully: {model_name}")
    
    
    with st.sidebar:
        st.header("📊 Model Dashboard")
        
        
        st.subheader("🔍 Model Information")
        st.info(f"📁 **File**: {model_name}")
        st.info(f"📐 **Input Shape**: {model.input_shape}")
        st.info(f"📊 **Output Shape**: {model.output_shape}")
        
        
        st.subheader("⚙️ Prediction Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.5, 0.1,
            help="Minimum confidence for high-confidence predictions"
        )
        
        show_probabilities = st.checkbox("Show All Probabilities", True)
        show_charts = st.checkbox("Show Prediction Charts", True)
        
        
        st.subheader("🏷️ Class Labels")
        num_classes = model.output_shape[-1] if len(model.output_shape) > 1 else 1
        
        if num_classes > 1:
            default_labels = [f"Class {i+1}" for i in range(num_classes)]
            class_labels_input = st.text_area(
                "Enter class labels (one per line):",
                value="\n".join(default_labels),
                height=100
            )
            class_labels = [label.strip() for label in class_labels_input.split('\n') if label.strip()]
        else:
            class_labels = ["Output"]
    
    
    st.header("📸 Image Upload & Prediction")
    

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image to get predictions from your AI model"
    )
    
    if uploaded_file is not None:
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
        
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)
            
            
            st.subheader("🔍 Image Information")
            st.write(f"**Size**: {image.size}")
            st.write(f"**Mode**: {image.mode}")
            st.write(f"**Format**: {image.format}")
        
        with col2:
            
            st.subheader("🎯 AI Prediction")
            
            if st.button("🚀 Predict", type="primary"):
                with st.spinner('🧠 AI is thinking...'):
                    try:
                        
                        processed_image = preprocess_image(image)
                        
                        
                        predictions = model.predict(processed_image, verbose=0)
                        
                        
                        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
                        
                        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                      
                            predicted_class_idx = np.argmax(predictions[0])
                            confidence = np.max(predictions[0])
                            predicted_class = class_labels[predicted_class_idx] if predicted_class_idx < len(class_labels) else f"Class {predicted_class_idx}"
                            
                           
                            st.markdown(f"### 🏆 Prediction Result")
                            st.markdown(f"**Predicted Class**: {predicted_class}")
                            
                           
                            confidence_class = "confidence-high" if confidence >= confidence_threshold else "confidence-low"
                            st.markdown(f'<p class="{confidence_class}">Confidence: {confidence:.2%}</p>', unsafe_allow_html=True)
                            
                            
                            if confidence >= confidence_threshold:
                                st.success("🎉 High Confidence Prediction!")
                            else:
                                st.warning("⚠️ Low Confidence - Consider more data")
                            
                           
                            st.progress(float(confidence))
                            
                        else:
                            
                            pred_value = predictions[0][0] if len(predictions.shape) > 1 else predictions[0]
                            st.markdown(f"### 📊 Prediction Value")
                            st.markdown(f"**Result**: {pred_value:.4f}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                       
                        st.subheader("📈 Detailed Analysis")
                        
                        if show_probabilities and len(predictions.shape) > 1 and predictions.shape[1] > 1:
                            
                            st.markdown("#### 🎯 All Class Probabilities")
                            
                            for i, (label, prob) in enumerate(zip(class_labels, predictions[0])):
                                col_label, col_prob, col_bar = st.columns([2, 1, 3])
                                
                                with col_label:
                                    st.write(f"**{label}**")
                                with col_prob:
                                    st.write(f"{prob:.3f}")
                                with col_bar:
                                    st.progress(float(prob))
                        
                       
                        if show_charts and len(predictions.shape) > 1 and predictions.shape[1] > 1:
                            st.markdown("#### 📊 Interactive Prediction Chart")
                            chart = create_prediction_chart(predictions, class_labels)
                            st.plotly_chart(chart, use_container_width=True)
                        
                        
                        st.subheader("📊 Prediction Statistics")
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Max Probability", f"{np.max(predictions[0]):.3f}")
                        
                        with col_stat2:
                            st.metric("Min Probability", f"{np.min(predictions[0]):.3f}")
                        
                        with col_stat3:
                            entropy = -np.sum(predictions[0] * np.log(predictions[0] + 1e-10))
                            st.metric("Prediction Entropy", f"{entropy:.3f}")
                        
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.info("💡 Try adjusting the image or check if the model is compatible")
    
    else:
       
        st.info("👆 Upload an image above to start making predictions!")
        
       
        st.subheader("🎨 How to Use")
        st.markdown("""
        1. **Upload an image** using the file uploader above
        2. **Adjust settings** in the sidebar if needed
        3. **Click 'Predict'** to get AI predictions
        4. **View results** with confidence scores and detailed analysis
        """)
        
       
        st.subheader("🔧 Model Capabilities")
        st.markdown(f"""
        - **Input Size**: {model.input_shape}
        - **Output Classes**: {model.output_shape[-1] if len(model.output_shape) > 1 else 1}
        - **Model Type**: {"Classification" if len(model.output_shape) > 1 and model.output_shape[-1] > 1 else "Regression/Binary"}
        """)
    
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 <b>Built with Streamlit & TensorFlow</b> 🚀</p>
        <p>Made with ❤️ for AI Model Deployment</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()