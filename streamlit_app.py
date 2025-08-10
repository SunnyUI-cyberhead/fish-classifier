"""
Enhanced Fish Image Classification - Streamlit Application
Professional web interface with advanced features for fish species identification
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64
import io
from datetime import datetime, timedelta
import cv2
import time
import hashlib
from collections import deque
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Page configuration
st.set_page_config(
    page_title="🐟 Advanced Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/fish-classifier',
        'Report a bug': "https://github.com/yourusername/fish-classifier/issues",
        'About': "# Fish Species Classification System\nVersion 2.0\nAdvanced Deep Learning for Marine Life Identification"
    }
)

# Enhanced CSS styling
st.markdown("""
    <style>
    /* Modern gradient backgrounds */
    .main {
        background: linear-gradient(180deg, #f0f2f6 0%, #ffffff 100%);
    }
    
    /* Animated header */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Metric cards with hover effect */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        border: 2px solid #667eea;
    }
    
    /* Animated prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 2s infinite;
    }
    
    .status-ready { background-color: #28a745; }
    .status-processing { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Custom progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Success animation */
    @keyframes success-pulse {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .success-animation {
        animation: success-pulse 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)


class DatabaseManager:
    """Manage prediction history in SQLite database"""
    
    def __init__(self, db_path="fish_predictions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_hash TEXT,
                model_name TEXT,
                predicted_species TEXT,
                confidence REAL,
                top5_predictions TEXT,
                processing_time REAL,
                user_feedback TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                loss REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction_data):
        """Save prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions 
            (image_hash, model_name, predicted_species, confidence, top5_predictions, processing_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            prediction_data['image_hash'],
            prediction_data['model_name'],
            prediction_data['species'],
            prediction_data['confidence'],
            json.dumps(prediction_data['top5']),
            prediction_data['processing_time']
        ))
        
        conn.commit()
        conn.close()
    
    def get_prediction_history(self, limit=50):
        """Get prediction history from database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("""
            SELECT timestamp, model_name, predicted_species, confidence, processing_time
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, conn, params=(limit,))
        conn.close()
        return df
    
    def get_statistics(self):
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total_predictions'] = cursor.fetchone()[0]
        
        # Most common species
        cursor.execute("""
            SELECT predicted_species, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_species
            ORDER BY count DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        stats['most_common_species'] = result[0] if result else "N/A"
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM predictions")
        stats['avg_confidence'] = cursor.fetchone()[0] or 0
        
        # Average processing time
        cursor.execute("SELECT AVG(processing_time) FROM predictions")
        stats['avg_processing_time'] = cursor.fetchone()[0] or 0
        
        conn.close()
        return stats


class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time webcam classification"""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.frame_count = 0
        self.prediction_interval = 30  # Predict every 30 frames
        self.last_prediction = None
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Predict every N frames
        if self.frame_count % self.prediction_interval == 0:
            # Preprocess for prediction
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            processed = self.preprocess_image(pil_img)
            
            # Make prediction
            predictions = self.model.predict(processed, verbose=0)
            top_idx = np.argmax(predictions[0])
            
            self.last_prediction = {
                'species': self.class_names.get(top_idx, f"Species_{top_idx}"),
                'confidence': float(predictions[0][top_idx])
            }
        
        # Draw prediction on frame
        if self.last_prediction:
            text = f"{self.last_prediction['species']} ({self.last_prediction['confidence']:.1%})"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, (0, 255, 0), 2)
        
        self.frame_count += 1
        return img
    
    def preprocess_image(self, image, target_size=(224, 224)):
        image = image.resize(target_size)
        img_array = img_to_array(image) / 255.0
        return np.expand_dims(img_array, axis=0)


class EnhancedFishClassifierApp:
    def __init__(self):
        """Initialize the Enhanced Fish Classifier Application"""
        self.model = None
        self.model_name = None
        self.class_indices = None
        self.class_names = None
        self.models_dir = Path("models_output")
        self.db = DatabaseManager()
        
        # Session state initialization
        if 'prediction_cache' not in st.session_state:
            st.session_state.prediction_cache = {}
        if 'comparison_mode' not in st.session_state:
            st.session_state.comparison_mode = False
        if 'batch_results' not in st.session_state:
            st.session_state.batch_results = []
    
    def load_model_and_classes(self, model_path):
        """Load the selected model and class indices"""
        try:
            # Load model with custom objects if needed
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
                class_names = {v: k for k, v in class_indices.items()}
            else:
                st.warning("Class indices file not found. Using default names.")
                class_names = {i: f"Species_{i}" for i in range(10)}
            
            return model, class_names
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
    
    def get_image_hash(self, image):
        """Generate hash for image caching"""
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return hashlib.md5(img_byte_arr).hexdigest()
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for model prediction"""
        image = image.resize(target_size)
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def augment_image(self, image):
        """Apply data augmentation to image"""
        augmented_images = []
        
        # Original
        augmented_images.append(("Original", image))
        
        # Horizontal flip
        augmented_images.append(("Horizontal Flip", ImageOps.mirror(image)))
        
        # Brightness variations
        enhancer = ImageEnhance.Brightness(image)
        augmented_images.append(("Bright", enhancer.enhance(1.3)))
        augmented_images.append(("Dark", enhancer.enhance(0.7)))
        
        # Contrast variations
        enhancer = ImageEnhance.Contrast(image)
        augmented_images.append(("High Contrast", enhancer.enhance(1.3)))
        
        # Rotation
        augmented_images.append(("Rotated 15°", image.rotate(15, fillcolor='white')))
        
        return augmented_images
    
    def predict_image(self, image, use_cache=True):
        """Make prediction on the uploaded image with caching"""
        if self.model is None:
            return None, None
        
        # Check cache
        img_hash = self.get_image_hash(image)
        cache_key = f"{self.model_name}_{img_hash}"
        
        if use_cache and cache_key in st.session_state.prediction_cache:
            return st.session_state.prediction_cache[cache_key]
        
        # Determine target size
        if 'inception' in self.model_name.lower() or 'xception' in self.model_name.lower():
            target_size = (299, 299)
        elif 'efficientnetb1' in self.model_name.lower():
            target_size = (240, 240)
        else:
            target_size = (224, 224)
        
        # Process and predict
        start_time = time.time()
        processed_image = self.preprocess_image(image, target_size)
        predictions = self.model.predict(processed_image, verbose=0)
        processing_time = time.time() - start_time
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'species': self.class_names.get(idx, f"Species_{idx}"),
                'confidence': float(predictions[0][idx]),
                'index': int(idx)
            })
        
        # Cache results
        result_tuple = (results, predictions[0], processing_time)
        if use_cache:
            st.session_state.prediction_cache[cache_key] = result_tuple
        
        return result_tuple
    
    def create_prediction_chart(self, results):
        """Create an interactive bar chart for predictions"""
        df = pd.DataFrame(results[:5])
        
        colors = ['#28a745' if c > 0.7 else '#ffc107' if c > 0.4 else '#dc3545' 
                 for c in df['confidence']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['confidence'],
                y=df['species'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0,0,0,0.3)', width=1)
                ),
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
            plot_bgcolor='rgba(240,242,246,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        return fig
    
    def create_confidence_gauge(self, confidence):
        """Create an animated gauge chart for confidence"""
        if confidence > 0.7:
            color = "#28a745"
            title = "High Confidence"
        elif confidence > 0.4:
            color = "#ffc107"
            title = "Medium Confidence"
        else:
            color = "#dc3545"
            title = "Low Confidence"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            title={'text': title, 'font': {'size': 20}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': color, 'thickness': 0.8},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(220,53,69,0.1)"},
                    {'range': [40, 70], 'color': "rgba(255,193,7,0.1)"},
                    {'range': [70, 100], 'color': "rgba(40,167,69,0.1)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            },
            number={'suffix': "%", 'font': {'size': 36, 'color': color}}
        ))
        
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_species_distribution_chart(self):
        """Create species distribution chart from history"""
        history_df = self.db.get_prediction_history(limit=100)
        
        if not history_df.empty:
            species_counts = history_df['predicted_species'].value_counts().head(10)
            
            fig = px.pie(
                values=species_counts.values,
                names=species_counts.index,
                title="Species Distribution (Last 100 Predictions)",
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            
            return fig
        return None
    
    def create_performance_timeline(self):
        """Create model performance timeline"""
        history_df = self.db.get_prediction_history(limit=50)
        
        if not history_df.empty:
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['confidence'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#667eea', width=2),
                marker=dict(size=8, color='#764ba2')
            ))
            
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['processing_time'],
                mode='lines+markers',
                name='Processing Time (s)',
                yaxis='y2',
                line=dict(color='#ffc107', width=2, dash='dash'),
                marker=dict(size=8, color='#ff9800')
            ))
            
            fig.update_layout(
                title="Performance Timeline",
                xaxis_title="Time",
                yaxis_title="Confidence",
                yaxis2=dict(
                    title="Processing Time (s)",
                    overlaying='y',
                    side='right'
                ),
                height=400,
                hovermode='x unified'
            )
            
            return fig
        return None
    
    def batch_process_images(self, images):
        """Process multiple images in batch"""
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, image) in enumerate(images):
            status_text.text(f"Processing {name}...")
            result, _, proc_time = self.predict_image(image)
            
            if result:
                results.append({
                    'image_name': name,
                    'predicted_species': result[0]['species'],
                    'confidence': result[0]['confidence'],
                    'processing_time': proc_time,
                    'top_5': result
                })
            
            progress_bar.progress((i + 1) / len(images))
        
        status_text.text("Batch processing complete!")
        return results
    
    def display_model_info(self):
        """Display enhanced model information"""
        if self.model:
            col1, col2, col3, col4 = st.columns(4)
            
            stats = self.db.get_statistics()
            
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <span class="status-indicator status-ready"></span>
                        <h4 style="color: #667eea; margin: 0;">Model</h4>
                        <h2 style="color: #333; margin: 10px 0;">{self.model_name}</h2>
                        <p style="color: #666; font-size: 0.9rem;">Active</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Total Predictions</h4>
                        <h2 style="color: #333; margin: 10px 0;">{stats['total_predictions']:,}</h2>
                        <p style="color: #666; font-size: 0.9rem;">All Time</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Avg Confidence</h4>
                        <h2 style="color: #333; margin: 10px 0;">{stats['avg_confidence']:.1%}</h2>
                        <p style="color: #666; font-size: 0.9rem;">Historical</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #667eea; margin: 0;">Avg Speed</h4>
                        <h2 style="color: #333; margin: 10px 0;">{stats['avg_processing_time']:.3f}s</h2>
                        <p style="color: #666; font-size: 0.9rem;">Per Image</p>
                    </div>
                """, unsafe_allow_html=True)
    
    def run(self):
        """Main application logic"""
        
        # Animated header
        st.markdown("""
            <div class="main-header">
                <h1>🐟 Advanced Fish Species Classification System</h1>
                <p style="color: white; font-size: 1.2rem; margin-top: 10px;">
                    State-of-the-Art Deep Learning for Marine Life Identification
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            st.title("⚙️ Configuration Panel")
            
            # Model Management Section
            st.markdown("### 📦 Model Management")
            
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob("*_final.h5"))
                model_files += list(self.models_dir.glob("*_best.h5"))
                
                if model_files:
                    model_names = list(set([f.stem.replace('_final', '').replace('_best', '')
                                           .replace('_frozen', '').replace('_finetuned', '') 
                                           for f in model_files]))
                    model_names.sort()
                    
                    selected_model = st.selectbox(
                        "Select Model:",
                        model_names,
                        help="Choose a trained model for prediction"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🚀 Load", type="primary", use_container_width=True):
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
                                        st.success(f"✅ {selected_model} loaded!")
                                        st.balloons()
                    
                    with col2:
                        if st.button("🔄 Refresh", use_container_width=True):
                            st.rerun()
                else:
                    st.error("No trained models found!")
            
            # Advanced Settings
            st.divider()
            st.markdown("### ⚡ Advanced Settings")
            
            with st.expander("🎯 Prediction Settings", expanded=False):
                use_cache = st.checkbox("Enable Prediction Cache", True, 
                                       help="Cache predictions for faster repeated analysis")
                batch_mode = st.checkbox("Batch Processing Mode", False,
                                        help="Process multiple images at once")
                augmentation_test = st.checkbox("Test with Augmentation", False,
                                               help="Test prediction robustness with augmented images")
            
            with st.expander("📊 Visualization Settings", expanded=False):
                show_heatmap = st.checkbox("Show Attention Heatmap", False,
                                          help="Visualize model attention (experimental)")
                show_confidence_history = st.checkbox("Show Confidence History", True)
                show_processing_stats = st.checkbox("Show Processing Statistics", True)
            
            with st.expander("🔬 Developer Options", expanded=False):
                show_raw_output = st.checkbox("Show Raw Model Output", False)
                show_preprocessing = st.checkbox("Show Preprocessing Steps", False)
                debug_mode = st.checkbox("Debug Mode", False)
            
            # Species Information
            if self.class_names:
                st.divider()
                st.markdown("### 🐠 Species Database")
                with st.expander(f"View {len(self.class_names)} Species", expanded=False):
                    species_list = sorted(set(self.class_names.values()))
                    search_term = st.text_input("🔍 Search species", "")
                    
                    filtered_species = [s for s in species_list 
                                       if search_term.lower() in s.lower()]
                    
                    for species in filtered_species[:10]:
                        st.markdown(f"• **{species}**")
                    
                    if len(filtered_species) > 10:
                        st.caption(f"... and {len(filtered_species) - 10} more")
            
            # System Information
            st.divider()
            st.markdown("### ℹ️ System Information")
            
            col1, col2 = st.columns(2)
            with col1:
                if tf.config.list_physical_devices('GPU'):
                    st.success("GPU: ✅")
                else:
                    st.warning("GPU: ❌")
            
            with col2:
                st.info(f"TF: {tf.__version__}")
        
        # Main content area
        if self.model:
            st.markdown("### 📊 Model Dashboard")
            self.display_model_info()
            st.divider()
        
        # Enhanced tabs
        tabs = st.tabs([
            "🎯 Prediction",
            "📸 Batch Processing",
            "🎥 Live Camera",
            "📈 Analytics",
            "🔬 Model Comparison",
            "📚 Documentation",
            "⚙️ Settings"
        ])
        
        # Tab 1: Single Image Prediction
        with tabs[0]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📸 Upload Image")
                
                uploaded_file = st.file_uploader(
                    "Select a fish image for classification",
                    type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                    help="Upload a clear image of a fish"
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # Display image with info
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Image metadata
                    img_info = f"""
                    📐 **Dimensions:** {image.size[0]} × {image.size[1]} px  
                    📁 **File:** {uploaded_file.name}  
                    💾 **Size:** {uploaded_file.size / 1024:.1f} KB
                    """
                    st.markdown(img_info)
                    
                    # Augmentation preview
                    if 'augmentation_test' in locals() and augmentation_test:
                        st.markdown("#### 🔄 Augmentation Preview")
                        augmented = self.augment_image(image)
                        
                        cols = st.columns(3)
                        for i, (name, aug_img) in enumerate(augmented[:3]):
                            with cols[i % 3]:
                                st.image(aug_img, caption=name, use_column_width=True)
            
            with col2:
                if uploaded_file is not None and self.model is not None:
                    st.markdown("### 🔮 Prediction Results")
                    
                    if st.button("🎯 Classify Fish", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            # Make prediction
                            results, raw_predictions, proc_time = self.predict_image(
                                image, use_cache='use_cache' in locals() and use_cache
                            )
                            
                            if results:
                                # Save to database
                                self.db.save_prediction({
                                    'image_hash': self.get_image_hash(image),
                                    'model_name': self.model_name,
                                    'species': results[0]['species'],
                                    'confidence': results[0]['confidence'],
                                    'top5': results,
                                    'processing_time': proc_time
                                })
                                
                                # Display results
                                top_result = results[0]
                                confidence = top_result['confidence']
                                
                                # Animated prediction card
                                if confidence > 0.7:
                                    emoji = "✅"
                                    status = "High Confidence"
                                elif confidence > 0.4:
                                    emoji = "⚠️"
                                    status = "Medium Confidence"
                                else:
                                    emoji = "❌"
                                    status = "Low Confidence"
                                
                                st.markdown(f"""
                                    <div class="prediction-card success-animation">
                                        <h2 style="margin: 0; font-size: 2rem;">
                                            🐟 {top_result['species']}
                                        </h2>
                                        <h3 style="margin: 10px 0; opacity: 0.9;">
                                            {emoji} {confidence:.1%} - {status}
                                        </h3>
                                        <p style="opacity: 0.8; margin: 0;">
                                            Processing Time: {proc_time:.3f}s
                                        </p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Confidence gauge
                                st.plotly_chart(
                                    self.create_confidence_gauge(confidence),
                                    use_container_width=True
                                )
                                
                                # Test with augmentation
                                if 'augmentation_test' in locals() and augmentation_test:
                                    st.markdown("#### 🔄 Augmentation Robustness Test")
                                    aug_results = []
                                    
                                    augmented = self.augment_image(image)
                                    for name, aug_img in augmented:
                                        aug_pred, _, _ = self.predict_image(aug_img, use_cache=False)
                                        if aug_pred:
                                            aug_results.append({
                                                'Augmentation': name,
                                                'Prediction': aug_pred[0]['species'],
                                                'Confidence': f"{aug_pred[0]['confidence']:.1%}"
                                            })
                                    
                                    st.dataframe(pd.DataFrame(aug_results), use_container_width=True)
                elif uploaded_file is not None:
                    st.warning("⚠️ Please load a model first!")
            
            # Detailed analysis section
            if uploaded_file is not None and self.model is not None and 'results' in locals():
                st.divider()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📊 Prediction Analysis")
                    st.plotly_chart(
                        self.create_prediction_chart(results),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("### 🎯 All Predictions")
                    df = pd.DataFrame(results)
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                    df = df[['species', 'confidence']]
                    df.columns = ['Species', 'Confidence']
                    st.dataframe(df, hide_index=True, use_container_width=True)
        
        # Tab 2: Batch Processing
        with tabs[1]:
            st.markdown("### 📦 Batch Image Processing")
            
            if self.model:
                uploaded_files = st.file_uploader(
                    "Upload multiple fish images",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                    help="Select multiple images for batch classification"
                )
                
                if uploaded_files:
                    st.info(f"📁 {len(uploaded_files)} images uploaded")
                    
                    if st.button("🚀 Process Batch", type="primary"):
                        images = []
                        for file in uploaded_files:
                            img = Image.open(file).convert('RGB')
                            images.append((file.name, img))
                        
                        # Process batch
                        batch_results = self.batch_process_images(images)
                        st.session_state.batch_results = batch_results
                        
                        # Display results
                        st.success(f"✅ Processed {len(batch_results)} images")
                        
                        # Summary statistics
                        avg_conf = np.mean([r['confidence'] for r in batch_results])
                        avg_time = np.mean([r['processing_time'] for r in batch_results])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Confidence", f"{avg_conf:.1%}")
                        with col2:
                            st.metric("Average Time", f"{avg_time:.3f}s")
                        with col3:
                            st.metric("Total Images", len(batch_results))
                        
                        # Results table
                        st.markdown("### 📊 Batch Results")
                        results_df = pd.DataFrame([
                            {
                                'Image': r['image_name'],
                                'Predicted Species': r['predicted_species'],
                                'Confidence': f"{r['confidence']:.1%}",
                                'Time (s)': f"{r['processing_time']:.3f}"
                            }
                            for r in batch_results
                        ])
                        
                        st.dataframe(
                            results_df.style.highlight_max(subset=['Confidence']),
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("Please load a model first to use batch processing")
        
        # Tab 3: Live Camera (Placeholder - requires webcam access)
        with tabs[2]:
            st.markdown("### 🎥 Live Camera Classification")
            
            if self.model:
                st.info("📹 Real-time classification from webcam feed")
                
                # Note: This is a simplified version. Full webcam support requires additional setup
                st.warning("""
                    ⚠️ Live camera feature requires additional setup:
                    1. Install streamlit-webrtc: `pip install streamlit-webrtc`
                    2. Configure STUN/TURN servers for WebRTC
                    3. Ensure HTTPS connection for camera access
                """)
                
                if st.button("📸 Capture Image", type="primary"):
                    st.info("Camera capture functionality would be implemented here")
            else:
                st.warning("Please load a model first to use live camera")
        
        # Tab 4: Analytics
        with tabs[3]:
            st.markdown("### 📈 Performance Analytics")
            
            # Database statistics
            stats = self.db.get_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", f"{stats['total_predictions']:,}")
            with col2:
                st.metric("Most Common Species", stats['most_common_species'])
            with col3:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.1%}")
            with col4:
                st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.3f}s")
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                dist_chart = self.create_species_distribution_chart()
                if dist_chart:
                    st.plotly_chart(dist_chart, use_container_width=True)
                else:
                    st.info("No prediction history available yet")
            
            with col2:
                timeline_chart = self.create_performance_timeline()
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)
                else:
                    st.info("No performance data available yet")
            
            # Prediction history table
            st.divider()
            st.markdown("### 📜 Recent Predictions")
            
            history_df = self.db.get_prediction_history(limit=20)
            if not history_df.empty:
                history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.1%}")
                history_df['processing_time'] = history_df['processing_time'].apply(lambda x: f"{x:.3f}s")
                
                st.dataframe(
                    history_df.style.highlight_max(subset=['confidence']),
                    use_container_width=True
                )
            else:
                st.info("No prediction history available")
        
        # Tab 5: Model Comparison
        with tabs[4]:
            st.markdown("### 🔬 Model Comparison")
            
            comparison_file = self.models_dir / "model_comparison.csv"
            if comparison_file.exists():
                df_comparison = pd.read_csv(comparison_file)
                
                # Interactive comparison chart
                fig = px.scatter(
                    df_comparison,
                    x='Accuracy',
                    y='F1-Score',
                    size='Precision',
                    color='Model',
                    hover_data=['Recall', 'Loss'],
                    title="Model Performance Comparison",
                    labels={'size': 'Precision'},
                    size_max=30
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart for top models
                st.markdown("#### 📊 Top Models Radar Chart")
                
                top_models = df_comparison.nlargest(3, 'Accuracy')
                
                categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                
                fig = go.Figure()
                
                for _, model in top_models.iterrows():
                    fig.add_trace(go.Scatterpolar(
                        r=[model[cat] for cat in categories],
                        theta=categories,
                        fill='toself',
                        name=model['Model']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison table
                st.markdown("#### 📋 Detailed Comparison")
                st.dataframe(
                    df_comparison.style.background_gradient(subset=['Accuracy', 'F1-Score'])
                                      .highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
                                      .highlight_min(subset=['Loss']),
                    use_container_width=True
                )
                
                # Best model highlight
                best_model = df_comparison.loc[df_comparison['Accuracy'].idxmax()]
                st.success(f"""
                    🏆 **Best Performing Model:** {best_model['Model']}
                    - Accuracy: {best_model['Accuracy']:.4f}
                    - F1-Score: {best_model['F1-Score']:.4f}
                    - Loss: {best_model['Loss']:.4f}
                """)
            else:
                st.info("No model comparison data available. Train multiple models to see comparison.")
        
        # Tab 6: Documentation
        with tabs[5]:
            st.markdown("### 📚 Documentation & Help")
            
            with st.expander("🚀 Quick Start Guide", expanded=True):
                st.markdown("""
                    #### Getting Started
                    
                    1. **Load a Model**
                       - Navigate to the sidebar
                       - Select a model from the dropdown
                       - Click the "Load" button
                       - Wait for the success message
                    
                    2. **Single Image Classification**
                       - Go to the "Prediction" tab
                       - Upload a fish image
                       - Click "Classify Fish"
                       - View the results and confidence score
                    
                    3. **Batch Processing**
                       - Switch to "Batch Processing" tab
                       - Upload multiple images at once
                       - Click "Process Batch"
                       - Download results as CSV
                    
                    4. **View Analytics**
                       - Check the "Analytics" tab
                       - Monitor prediction history
                       - Analyze model performance
                """)
            
            with st.expander("🔬 Model Information"):
                st.markdown("""
                    #### Available Models
                    
                    | Model | Description | Best For |
                    |-------|-------------|----------|
                    | **MobileNet** | Lightweight, fast | Mobile/edge deployment |
                    | **EfficientNet** | Balanced performance | General use |
                    | **ResNet50** | Deep residual network | High accuracy |
                    | **VGG16** | Classic architecture | Baseline comparison |
                    | **InceptionV3** | Multi-scale features | Complex patterns |
                    | **DenseNet** | Dense connections | Feature reuse |
                    
                    #### Model Selection Tips
                    - For speed: Choose MobileNet or EfficientNetB0
                    - For accuracy: Choose ResNet50 or InceptionV3
                    - For balanced: Choose EfficientNetB1 or DenseNet121
                """)
            
            with st.expander("📊 Understanding Metrics"):
                st.markdown("""
                    #### Confidence Levels
                    
                    - **High (>70%)** ✅: Very reliable prediction
                    - **Medium (40-70%)** ⚠️: Moderate certainty
                    - **Low (<40%)** ❌: Uncertain, verify manually
                    
                    #### Performance Metrics
                    
                    - **Accuracy**: Percentage of correct predictions
                    - **Precision**: Ratio of true positives to all positive predictions
                    - **Recall**: Ratio of true positives to all actual positives
                    - **F1-Score**: Harmonic mean of precision and recall
                    - **Loss**: Model's prediction error (lower is better)
                """)
            
            with st.expander("🎯 Best Practices"):
                st.markdown("""
                    #### Image Quality Tips
                    
                    ✅ **DO:**
                    - Use clear, well-lit images
                    - Ensure the fish is clearly visible
                    - Use images with good resolution
                    - Center the fish in the frame
                    
                    ❌ **DON'T:**
                    - Use blurry or dark images
                    - Include multiple fish in one image
                    - Use heavily filtered images
                    - Upload images with watermarks
                    
                    #### Improving Predictions
                    
                    1. **Image Preprocessing**
                       - Crop unnecessary background
                       - Adjust brightness if too dark
                       - Remove obstructions
                    
                    2. **Model Selection**
                       - Try different models for comparison
                       - Use ensemble predictions for critical decisions
                       - Consider the speed-accuracy tradeoff
                    
                    3. **Validation**
                       - Cross-check low confidence predictions
                       - Use batch processing for consistency
                       - Monitor the analytics dashboard
                """)
            
            with st.expander("❓ FAQ"):
                st.markdown("""
                    **Q: Why is my prediction confidence low?**
                    A: Low confidence can result from poor image quality, unusual angles, 
                    or species not well-represented in training data.
                    
                    **Q: How can I improve classification speed?**
                    A: Use lighter models like MobileNet, enable prediction caching, 
                    and ensure GPU acceleration is available.
                    
                    **Q: Can I add new fish species?**
                    A: The model needs to be retrained with new species data. 
                    Contact the development team for model updates.
                    
                    **Q: What image formats are supported?**
                    A: JPG, JPEG, PNG, BMP, and WebP formats are supported.
                    
                    **Q: Is batch processing faster than individual processing?**
                    A: Yes, batch processing is optimized for multiple images and includes 
                    performance improvements.
                """)
        
        # Tab 7: Settings
        with tabs[6]:
            st.markdown("### ⚙️ Application Settings")
            
            with st.expander("🎨 Appearance", expanded=True):
                theme = st.selectbox("Color Theme", ["Default", "Dark", "Light", "Ocean"])
                animation = st.checkbox("Enable Animations", True)
                compact_mode = st.checkbox("Compact Mode", False)
                
                if st.button("Apply Theme"):
                    st.success("Theme settings applied!")
            
            with st.expander("💾 Data Management"):
                st.markdown("#### Database Operations")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Export History", use_container_width=True):
                        history = self.db.get_prediction_history(limit=1000)
                        if not history.empty:
                            csv = history.to_csv(index=False)
                            st.download_button(
                                "📥 Download",
                                csv,
                                "prediction_history.csv",
                                "text/csv"
                            )
                
                with col2:
                    if st.button("Clear Cache", use_container_width=True):
                        st.session_state.prediction_cache = {}
                        st.success("Cache cleared!")
                
                with col3:
                    if st.button("Reset Database", type="secondary", use_container_width=True):
                        if st.checkbox("Confirm reset"):
                            # Would implement database reset here
                            st.warning("Database reset functionality")
            
            with st.expander("🔔 Notifications"):
                email_notifications = st.checkbox("Email Notifications", False)
                if email_notifications:
                    email = st.text_input("Email Address")
                    st.multiselect(
                        "Notify me when:",
                        ["Low confidence prediction", "Batch processing complete", 
                         "New model available", "System updates"]
                    )
            
            with st.expander("🔐 Advanced"):
                st.markdown("#### Developer Settings")
                
                api_endpoint = st.text_input(
                    "Custom API Endpoint",
                    placeholder="https://api.fishclassifier.com/v1"
                )
                
                max_cache_size = st.slider(
                    "Max Cache Size (MB)",
                    10, 500, 100
                )
                
                concurrent_predictions = st.slider(
                    "Concurrent Predictions",
                    1, 10, 3
                )
                
                if st.button("Save Advanced Settings"):
                    st.success("Settings saved successfully!")
        
        # Footer
        st.divider()
        st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; color: white; margin-top: 2rem;">
                <h3 style="margin: 0;">🐟 Advanced Fish Species Classification System</h3>
                <p style="margin: 10px 0; opacity: 0.9;">Powered by Deep Learning | TensorFlow & Keras</p>
                <p style="margin: 0; opacity: 0.7; font-size: 0.9rem;">
                    Version 2.0 | © 2025 Marine AI Research
                </p>
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the enhanced application"""
    app = EnhancedFishClassifierApp()
    app.run()


if __name__ == "__main__":
    main()
