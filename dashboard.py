"""
Fish Species Classification - Dark Theme Professional App
Fixed classification and improved visibility
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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fish Species Classifier",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Dark theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.4rem;
        margin-top: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #2d3561 0%, #353b64 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102,126,234,0.4);
        border-color: #667eea;
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Tabs with dark theme */
    .stTabs {
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(45,53,97,0.5);
        border-radius: 15px;
        padding: 10px;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
        background: rgba(53,59,100,0.8);
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 12px 24px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102,126,234,0.3);
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff !important;
        border: none;
    }
    
    /* Tab panels dark background */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(26,26,46,0.6);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 12px 30px;
        border: none;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px rgba(102,126,234,0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102,126,234,0.6);
    }
    
    /* Sidebar dark theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(102,126,234,0.3);
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(45,53,97,0.5);
        border: 2px dashed #667eea;
        border-radius: 15px;
        color: #ffffff;
    }
    
    /* Success/Warning/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #f09819, #edde5d);
        color: #1a1a2e;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #f85032, #e73827);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(56,239,125,0.3);
        margin: 2rem 0;
    }
    
    .prediction-card h2 {
        color: #1a1a2e;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .prediction-card h3 {
        color: #1a1a2e;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* DataFrame dark theme */
    [data-testid="stDataFrame"] {
        background: rgba(45,53,97,0.8);
        border-radius: 10px;
        color: white;
    }
    
    /* Expander dark theme */
    .streamlit-expanderHeader {
        background: rgba(53,59,100,0.8);
        color: white !important;
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 10px;
        font-weight: 600;
    }
    
    [data-testid="stExpander"] {
        background: rgba(26,26,46,0.6);
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 10px;
    }
    
    /* Text color fixes */
    .stMarkdown, .stText, p, span, label {
        color: #ffffff !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(45,53,97,0.5);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(102,126,234,0.3);
    }
    
    [data-testid="metric-container"] label {
        color: #a0a0a0 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background: rgba(45,53,97,0.8);
        color: white;
        border: 1px solid rgba(102,126,234,0.3);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_classes(model_path, models_dir):
    """Load model and class indices with caching"""
    try:
        # Load the model
        model = load_model(model_path)
        
        # Load class indices
        class_indices_path = models_dir / 'class_indices.json'
        if class_indices_path.exists():
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            class_names = {v: k for k, v in class_indices.items()}
        else:
            class_names = {i: f"Fish_Species_{i+1}" for i in range(11)}
        
        return model, class_names
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None


def preprocess_image(image, model_name):
    """Preprocess image for prediction"""
    # Determine target size
    if 'inception' in model_name.lower() or 'xception' in model_name.lower():
        target_size = (299, 299)
    elif 'efficientnetb1' in model_name.lower():
        target_size = (240, 240)
    else:
        target_size = (224, 224)
    
    # Preprocess
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


def create_gauge_chart(confidence):
    """Create gauge chart with dark theme"""
    color = "#38ef7d" if confidence > 0.7 else "#f09819" if confidence > 0.4 else "#f85032"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        title = {'text': "Confidence Level", 'font': {'size': 24, 'color': '#ffffff'}},
        number = {'suffix': "%", 'font': {'size': 36, 'color': '#ffffff'}},
        delta = {'reference': 70, 'increasing': {'color': "#38ef7d"}, 'decreasing': {'color': "#f85032"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#ffffff"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(45,53,97,0.5)",
            'borderwidth': 2,
            'bordercolor': "#667eea",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(248,80,50,0.3)'},
                {'range': [40, 70], 'color': 'rgba(240,152,25,0.3)'},
                {'range': [70, 100], 'color': 'rgba(56,239,125,0.3)'}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=60, b=30),
        paper_bgcolor='rgba(26,26,46,0)',
        plot_bgcolor='rgba(26,26,46,0)',
        font={'color': '#ffffff'}
    )
    
    return fig


def create_bar_chart(results):
    """Create bar chart with dark theme"""
    df = pd.DataFrame(results[:5])
    
    colors = ['#38ef7d' if c > 0.7 else '#f09819' if c > 0.4 else '#f85032' 
             for c in df['confidence']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['confidence'],
            y=df['species'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#667eea', width=2)
            ),
            text=[f"{c:.1%}" for c in df['confidence']],
            textposition='outside',
            textfont=dict(size=14, color='#ffffff')
        )
    ])
    
    fig.update_layout(
        title={'text': "Top 5 Predictions", 'font': {'size': 24, 'color': '#ffffff'}},
        xaxis_title="Confidence Score",
        yaxis_title="Fish Species",
        xaxis=dict(
            tickformat='.0%',
            range=[0, max(df['confidence'].max() * 1.2, 0.1)],
            gridcolor='rgba(102,126,234,0.2)',
            tickfont=dict(color='#ffffff')
        ),
        yaxis=dict(
            gridcolor='rgba(102,126,234,0.2)',
            tickfont=dict(color='#ffffff')
        ),
        height=450,
        margin=dict(l=0, r=100, t=60, b=0),
        plot_bgcolor='rgba(26,26,46,0.5)',
        paper_bgcolor='rgba(26,26,46,0)',
        font={'color': '#ffffff'}
    )
    
    return fig


def main():
    """Main application"""
    
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🐟 Fish Species Classification</h1>
            <p>Advanced AI-Powered Marine Life Identification</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ **Configuration**")
        st.markdown("---")
        
        st.markdown("### 📦 **Model Selection**")
        
        models_dir = Path("models_output")
        if models_dir.exists():
            model_files = list(models_dir.glob("*_final.h5")) + list(models_dir.glob("*_best.h5"))
            
            if model_files:
                model_names = list(set([
                    f.stem.replace('_final', '').replace('_best', '')
                    .replace('_frozen', '').replace('_finetuned', '') 
                    for f in model_files
                ]))
                model_names.sort()
                
                selected_model = st.selectbox(
                    "Choose Model:",
                    model_names,
                    key="model_selector"
                )
                
                if st.button("🚀 **Load Model**", type="primary"):
                    model_file = None
                    for suffix in ['_final.h5', '_finetuned_best.h5', '_frozen_best.h5', '_best.h5']:
                        potential_file = models_dir / f"{selected_model}{suffix}"
                        if potential_file.exists():
                            model_file = potential_file
                            break
                    
                    if model_file:
                        with st.spinner(f"Loading {selected_model}..."):
                            model, class_names = load_model_and_classes(model_file, models_dir)
                            if model:
                                st.session_state.model = model
                                st.session_state.class_names = class_names
                                st.session_state.model_name = selected_model
                                st.session_state.model_loaded = True
                                st.success(f"✅ **{selected_model} loaded successfully!**")
        
        if st.session_state.model_loaded:
            st.markdown("---")
            st.info(f"**Model:** {st.session_state.model_name}")
        
        if st.session_state.class_names:
            st.markdown("---")
            st.markdown("### 🐠 **Fish Species**")
            with st.expander("View all species"):
                for i, species in enumerate(sorted(set(st.session_state.class_names.values())), 1):
                    st.write(f"{i}. {species}")
        
        st.markdown("---")
        st.markdown("### ⚡ **Settings**")
        show_preprocessing = st.checkbox("Show Preprocessing Steps")
        show_raw = st.checkbox("Show Raw Predictions")
    
    # Model Information
    if st.session_state.model_loaded:
        st.markdown("### 📊 **Model Information**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Model</div>
                    <div class="metric-value">{}</div>
                </div>
            """.format(st.session_state.model_name), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Classes</div>
                    <div class="metric-value">{}</div>
                </div>
            """.format(len(st.session_state.class_names)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Parameters</div>
                    <div class="metric-value">{:,}</div>
                </div>
            """.format(st.session_state.model.count_params()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class="metric-card">
                    <div class="metric-label">Status</div>
                    <div class="metric-value" style="color: #38ef7d;">Ready</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 **Prediction**", "📈 **Analysis**", "📚 **Documentation**"])
    
    with tab1:
        if not st.session_state.model_loaded:
            st.warning("⚠️ **Please load a model first!**")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📸 **Upload Image**")
                
                uploaded_file = st.file_uploader(
                    "Choose a fish image...",
                    type=['jpg', 'jpeg', 'png'],
                    key="uploader"
                )
                
                if uploaded_file:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Width", f"{image.size[0]} px")
                    with col_b:
                        st.metric("Height", f"{image.size[1]} px")
            
            with col2:
                st.markdown("### 🔮 **Prediction Results**")
                
                if uploaded_file and st.session_state.model_loaded:
                    if st.button("🎯 **Classify Fish**", type="primary", key="classify"):
                        with st.spinner("🔄 Analyzing..."):
                            try:
                                # Preprocess image
                                processed_img = preprocess_image(image, st.session_state.model_name)
                                
                                # Make prediction
                                predictions = st.session_state.model.predict(processed_img, verbose=0)
                                
                                # Get top 5
                                top_indices = np.argsort(predictions[0])[-5:][::-1]
                                
                                results = []
                                for idx in top_indices:
                                    results.append({
                                        'species': st.session_state.class_names.get(idx, f"Species_{idx+1}"),
                                        'confidence': float(predictions[0][idx])
                                    })
                                
                                st.session_state.prediction_results = results
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                    
                    # Display results
                    if st.session_state.prediction_results:
                        top = st.session_state.prediction_results[0]
                        conf = top['confidence']
                        
                        if conf > 0.7:
                            status = "✅ High Confidence"
                        elif conf > 0.4:
                            status = "⚠️ Medium Confidence"
                        else:
                            status = "❌ Low Confidence"
                        
                        st.markdown(f"""
                            <div class="prediction-card">
                                <h2>🐟 {top['species']}</h2>
                                <h3>{conf:.1%} - {status}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if show_raw:
                            with st.expander("📋 Raw Predictions"):
                                st.json(st.session_state.prediction_results)
                else:
                    st.info("👆 **Upload an image to classify!**")
            
            # Detailed Analysis
            if st.session_state.prediction_results:
                st.markdown("---")
                st.markdown("### 📊 **Detailed Analysis**")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.plotly_chart(
                        create_gauge_chart(st.session_state.prediction_results[0]['confidence']),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("#### **All Predictions**")
                    df = pd.DataFrame(st.session_state.prediction_results)
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                    df.columns = ['Species', 'Confidence']
                    st.dataframe(df, use_container_width=True)
                
                st.plotly_chart(
                    create_bar_chart(st.session_state.prediction_results),
                    use_container_width=True
                )
    
    with tab2:
        st.markdown("### 📈 **Model Performance Analysis**")
        
        comparison_file = models_dir / "model_comparison.csv" if 'models_dir' in locals() else Path("models_output/model_comparison.csv")
        
        if comparison_file.exists():
            df_comp = pd.read_csv(comparison_file)
            
            fig = px.bar(
                df_comp,
                x='Model',
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                title="Model Performance Comparison",
                barmode='group',
                height=500,
                color_discrete_sequence=['#667eea', '#764ba2', '#38ef7d', '#f09819']
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(26,26,46,0.5)',
                paper_bgcolor='rgba(26,26,46,0)',
                font={'color': '#ffffff'},
                title_font_size=24
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            best = df_comp.loc[df_comp['Accuracy'].idxmax()]
            
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 2rem; border-radius: 15px; text-align: center;">
                    <h3 style="color: #1a1a2e;">🏆 Best Model: {best['Model']}</h3>
                    <p style="color: #1a1a2e; font-size: 1.2rem;">
                        Accuracy: {best['Accuracy']:.2%} | F1-Score: {best['F1-Score']:.4f}
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.info("📊 **No comparison data found.**")
    
    with tab3:
        st.markdown("### 📚 **Documentation**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("🚀 **Quick Start**", expanded=True):
                st.markdown("""
                #### Steps:
                1. Load a model from sidebar
                2. Upload fish image
                3. Click Classify Fish
                4. View results
                
                #### Tips:
                - Use clear images
                - Good lighting helps
                - Center the fish
                """)
            
            with st.expander("🔬 **Models**"):
                st.markdown("""
                - **MobileNet** - Fast
                - **EfficientNet** - Accurate
                - **ResNet** - Reliable
                - **VGG** - Classic
                """)
        
        with col2:
            with st.expander("📊 **Metrics**"):
                st.markdown("""
                #### Confidence:
                - 70-100% = High ✅
                - 40-70% = Medium ⚠️
                - 0-40% = Low ❌
                """)
            
            with st.expander("🛠️ **Troubleshooting**"):
                st.markdown("""
                - Load model first
                - Check image format
                - Try different image
                - Restart if needed
                """)


if __name__ == "__main__":
    main()