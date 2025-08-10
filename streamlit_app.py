"""
Fish Image Classification - Streamlit Deployment Application
Professional web interface for real-time fish species prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64
import io
from datetime import datetime
import cv2

# Page configuration
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        height: 100%;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 5px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: white;
        border-radius: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


class FishClassifierApp:
    def __init__(self):
        """Initialize the Fish Classifier Application"""
        self.model = None
        self.model_name = None
        self.class_indices = None
        self.class_names = None
        self.models_dir = Path("models_output")
        self.prediction_history = []
        
    def load_model_and_classes(self, model_path):
        """Load the selected model and class indices"""
        try:
            # Load model
            model = load_model(model_path, compile=False)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load class indices
            class_indices_path = self.models_dir / 'class_indices.json'
            if class_indices_path.exists():
                with open(class_indices_path, 'r') as f:
                    class_indices = json.load(f)
                # Reverse mapping for predictions
                class_names = {v: k for k, v in class_indices.items()}
            else:
                # Default class names if file not found
                st.warning("Class indices file not found. Using default names.")
                class_names = {i: f"Species_{i}" for i in range(10)}
            
            return model, class_names
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for model prediction"""
        # Resize image
        image = image.resize(target_size)
        
        # Convert to array and normalize
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_image(self, image):
        """Make prediction on the uploaded image"""
        if self.model is None:
            return None, None
        
        # Determine target size based on model
        if 'inception' in self.model_name.lower() or 'xception' in self.model_name.lower():
            target_size = (299, 299)
        elif 'efficientnetb1' in self.model_name.lower():
            target_size = (240, 240)
        else:
            target_size = (224, 224)
        
        # Preprocess image
        processed_image = self.preprocess_image(image, target_size)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'species': self.class_names.get(idx, f"Species_{idx}"),
                'confidence': float(predictions[0][idx]),
                'index': int(idx)
            })
        
        return results, predictions[0]
    
    def create_prediction_chart(self, results):
        """Create an interactive bar chart for predictions"""
        df = pd.DataFrame(results[:5])
        
        # Create color scale based on confidence
        colors = ['#28a745' if c > 0.7 else '#ffc107' if c > 0.4 else '#dc3545' 
                 for c in df['confidence']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['confidence'],
                y=df['species'],
                orientation='h',
                marker=dict(color=colors),
                text=[f"{c:.1%}" for c in df['confidence']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2%}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Top 5 Predictions",
            xaxis_title="Confidence Score",
            yaxis_title="Fish Species",
            xaxis=dict(tickformat='.0%', range=[0, max(df['confidence'].max() * 1.1, 0.1)]),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_confidence_gauge(self, confidence):
        """Create a gauge chart for confidence visualization"""
        # Determine color based on confidence
        if confidence > 0.7:
            color = "#28a745"
        elif confidence > 0.4:
            color = "#ffc107"
        else:
            color = "#dc3545"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            title={'text': "Confidence Level"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 70], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            },
            number={'suffix': "%", 'font': {'size': 24}}
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def display_model_info(self):
        """Display information about the loaded model"""
        if self.model:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #667eea;">Model</h4>
                        <h2 style="color: #333;">{}</h2>
                    </div>
                """.format(self.model_name), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #667eea;">Classes</h4>
                        <h2 style="color: #333;">{}</h2>
                    </div>
                """.format(len(self.class_names)), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #667eea;">Parameters</h4>
                        <h2 style="color: #333;">{:,}</h2>
                    </div>
                """.format(self.model.count_params()), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                    <div class="metric-card">
                        <h4 style="color: #667eea;">Status</h4>
                        <h2 style="color: #28a745;">Ready</h2>
                    </div>
                """.format(), unsafe_allow_html=True)
    
    def run(self):
        """Main application logic"""
        
        # Header
        st.markdown("""
            <div class="main-header">
                <h1>🐟 Fish Species Classification System</h1>
                <p>Advanced Deep Learning Models for Marine Life Identification</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # Model selection
            st.subheader("📦 Model Selection")
            
            # Get available models
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob("*_final.h5"))
                model_files += list(self.models_dir.glob("*_best.h5"))
                
                if model_files:
                    # Remove duplicates and sort
                    model_names = list(set([f.stem.replace('_final', '').replace('_best', '').replace('_frozen', '').replace('_finetuned', '') 
                                           for f in model_files]))
                    model_names.sort()
                    
                    selected_model = st.selectbox(
                        "Choose Model:",
                        model_names,
                        help="Select a trained model for prediction"
                    )
                    
                    if st.button("🚀 Load Model", type="primary"):
                        # Find the best available file for selected model
                        model_file = None
                        for suffix in ['_final.h5', '_finetuned_best.h5', '_frozen_best.h5', '_best.h5']:
                            potential_file = self.models_dir / f"{selected_model}{suffix}"
                            if potential_file.exists():
                                model_file = potential_file
                                break
                        
                        if model_file:
                            with st.spinner(f"Loading {selected_model}..."):
                                self.model, self.class_names = self.load_model_and_classes(model_file)
                                if self.model:
                                    self.model_name = selected_model
                                    st.success(f"✅ {selected_model} loaded successfully!")
                else:
                    st.error("No trained models found!")
                    st.info("Please train models first using the training scripts.")
            else:
                st.error("Models directory not found!")
                st.info("Expected directory: models_output/")
            
            # Display available classes
            if self.class_names:
                st.divider()
                st.subheader("🐠 Fish Species")
                with st.expander("View all species", expanded=False):
                    species_list = sorted(set(self.class_names.values()))
                    for species in species_list:
                        st.write(f"• {species}")
            
            # Settings
            st.divider()
            st.subheader("⚡ Settings")
            show_preprocessing = st.checkbox("Show Preprocessing Steps", False)
            show_raw_output = st.checkbox("Show Raw Predictions", False)
            
            # Info
            st.divider()
            st.subheader("ℹ️ About")
            st.info("""
                This application uses deep learning models to classify fish species.
                
                **How to use:**
                1. Load a model
                2. Upload an image
                3. View predictions
            """)
        
        # Main content area
        if self.model:
            # Display model info
            st.subheader("📊 Model Information")
            self.display_model_info()
            
            st.divider()
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Analysis", "📚 Documentation"])
        
        with tab1:
            # Prediction interface
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Upload Image")
                
                uploaded_file = st.file_uploader(
                    "Choose a fish image...",
                    type=['jpg', 'jpeg', 'png'],
                    help="Upload a clear image of a fish for classification"
                )
                
                if uploaded_file is not None:
                    # Display uploaded image
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Image info
                    st.caption(f"📐 Original size: {image.size[0]} × {image.size[1]} pixels")
                    st.caption(f"📁 File: {uploaded_file.name}")
                    
                    # Show preprocessing if enabled
                    if show_preprocessing and 'show_preprocessing' in locals():
                        with st.expander("Preprocessing Steps"):
                            st.write("1. Convert to RGB")
                            st.write("2. Resize to model input size")
                            st.write("3. Normalize pixel values to [0, 1]")
                            st.write("4. Add batch dimension")
            
            with col2:
                if uploaded_file is not None and self.model is not None:
                    st.subheader("🔮 Prediction Results")
                    
                    # Make prediction
                    if st.button("🎯 Classify Fish", type="primary"):
                        with st.spinner("Analyzing image..."):
                            # Predict
                            results, raw_predictions = self.predict_image(image)
                            
                            if results:
                                # Top prediction
                                top_result = results[0]
                                confidence = top_result['confidence']
                                
                                # Determine confidence level
                                if confidence > 0.7:
                                    confidence_class = "confidence-high"
                                    confidence_text = "High Confidence"
                                    emoji = "✅"
                                elif confidence > 0.4:
                                    confidence_class = "confidence-medium"
                                    confidence_text = "Medium Confidence"
                                    emoji = "⚠️"
                                else:
                                    confidence_class = "confidence-low"
                                    confidence_text = "Low Confidence"
                                    emoji = "❌"
                                
                                # Display prediction
                                st.markdown(f"""
                                    <div class="prediction-card">
                                        <h2 style="color: #333; margin: 0;">🐟 {top_result['species']}</h2>
                                        <h3 class="{confidence_class}">
                                            {emoji} {confidence:.1%} - {confidence_text}
                                        </h3>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Confidence gauge
                                st.plotly_chart(
                                    self.create_confidence_gauge(confidence),
                                    use_container_width=True
                                )
                                
                                # Store in history
                                self.prediction_history.append({
                                    'timestamp': datetime.now(),
                                    'species': top_result['species'],
                                    'confidence': confidence,
                                    'model': self.model_name
                                })
                                
                                # Show raw predictions if enabled
                                if show_raw_output and 'show_raw_output' in locals():
                                    with st.expander("Raw Predictions"):
                                        st.json(results)
                elif uploaded_file is not None and self.model is None:
                    st.warning("⚠️ Please load a model first!")
            
            # Detailed predictions
            if uploaded_file is not None and self.model is not None and 'results' in locals():
                st.divider()
                st.subheader("📊 Detailed Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bar chart of predictions
                    st.plotly_chart(
                        self.create_prediction_chart(results),
                        use_container_width=True
                    )
                
                with col2:
                    # Prediction table
                    st.write("**All Predictions:**")
                    df = pd.DataFrame(results)
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                    df = df[['species', 'confidence']]
                    df.columns = ['Species', 'Confidence']
                    st.dataframe(df, hide_index=True, use_container_width=True)
        
        with tab2:
            # Analysis tab
            st.subheader("📈 Model Performance Analysis")
            
            # Load and display model comparison if available
            comparison_file = self.models_dir / "model_comparison.csv"
            if comparison_file.exists():
                df_comparison = pd.read_csv(comparison_file)
                
                # Performance metrics
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create comparison chart
                    fig = px.bar(
                        df_comparison,
                        x='Model',
                        y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        title="Model Performance Comparison",
                        barmode='group',
                        height=400
                    )
                    fig.update_layout(
                        xaxis_title="Model",
                        yaxis_title="Score",
                        legend_title="Metric"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Best model info
                    best_model = df_comparison.loc[df_comparison['Accuracy'].idxmax()]
                    st.markdown("""
                        <div class="success-box">
                            <h4>🏆 Best Model</h4>
                            <p><strong>{}</strong></p>
                            <p>Accuracy: {:.2%}</p>
                            <p>F1-Score: {:.4f}</p>
                        </div>
                    """.format(best_model['Model'], best_model['Accuracy'], best_model['F1-Score']), 
                    unsafe_allow_html=True)
                
                # Model comparison table
                st.subheader("📊 Detailed Metrics")
                st.dataframe(
                    df_comparison.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
                                      .highlight_min(axis=0, subset=['Loss']),
                    use_container_width=True
                )
            else:
                st.info("No model comparison data found. Train models first to see analysis.")
            
            # Prediction history
            if self.prediction_history:
                st.divider()
                st.subheader("📜 Prediction History")
                
                history_df = pd.DataFrame(self.prediction_history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(
                    history_df[['timestamp', 'species', 'confidence', 'model']],
                    use_container_width=True,
                    hide_index=True
                )
        
        with tab3:
            # Documentation tab
            st.subheader("📚 Documentation")
            
            with st.expander("🎯 How to Use", expanded=True):
                st.markdown("""
                    ### Quick Start Guide
                    
                    1. **Load a Model**
                       - Select a model from the sidebar dropdown
                       - Click "Load Model" button
                       - Wait for confirmation message
                    
                    2. **Upload an Image**
                       - Click "Browse files" or drag and drop
                       - Supported formats: JPG, JPEG, PNG
                       - Best results with clear, well-lit images
                    
                    3. **Get Predictions**
                       - Click "Classify Fish" button
                       - View species prediction and confidence
                       - Check detailed analysis below
                """)
            
            with st.expander("🔬 Model Information"):
                st.markdown("""
                    ### Available Models
                    
                    - **MobileNet**: Lightweight, fast inference
                    - **EfficientNet**: Best accuracy/efficiency ratio
                    - **ResNet**: Deep residual learning
                    - **VGG**: Classic architecture
                    - **InceptionV3**: Multi-scale features
                    - **DenseNet**: Dense connections
                    
                    ### Preprocessing
                    - Images resized to model input size
                    - Pixel values normalized to [0, 1]
                    - Data augmentation during training
                """)
            
            with st.expander("📊 Understanding Results"):
                st.markdown("""
                    ### Confidence Levels
                    
                    - **High (>70%)**: Very confident prediction ✅
                    - **Medium (40-70%)**: Moderate confidence ⚠️
                    - **Low (<40%)**: Uncertain prediction ❌
                    
                    ### Metrics Explained
                    
                    - **Accuracy**: Overall correct predictions
                    - **Precision**: Correct positive predictions
                    - **Recall**: Found all positive cases
                    - **F1-Score**: Balance of precision and recall
                """)
            
            with st.expander("🛠️ Troubleshooting"):
                st.markdown("""
                    ### Common Issues
                    
                    **Model not loading?**
                    - Check models_output directory exists
                    - Ensure model files are present (.h5)
                    - Try restarting the application
                    
                    **Low confidence predictions?**
                    - Use clear, well-lit images
                    - Ensure fish is clearly visible
                    - Try different angles or images
                    
                    **Application slow?**
                    - First prediction takes longer (model loading)
                    - Larger models (VGG, ResNet) are slower
                    - Consider using MobileNet for speed
                """)
        
        # Footer
        st.divider()
        st.markdown("""
            <div style="text-align: center; color: #666; padding: 1rem;">
                <p>🐟 Fish Species Classification System | Powered by Deep Learning</p>
                <p>Built with TensorFlow, Keras, and Streamlit</p>
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the application"""
    app = FishClassifierApp()
    app.run()


if __name__ == "__main__":
    main()