import streamlit as st

st.set_page_config(
    page_title="AgriLeaf Analytics | Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# LAZY LOADING - Don't load model at startup, load on first use to prevent freeze
model = None

def get_model():
    global model
    if model is None:
        with st.spinner("Loading neural network weights... (one-time)"):
            model = load_model('weights/plant_disease_classifier.h5')
    return model

def predict_disease(image_file):
    img = Image.open(image_file)
    img = img.resize((256, 256))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = get_model().predict(img_array, verbose=0)
    labels = ['Healthy', 'Powdery Mildew', 'Rust']
    
    pred_idx = np.argmax(preds[0])
    return labels[pred_idx], float(preds[0][pred_idx]), dict(zip(labels, [float(x) for x in preds[0]]))

# MODERN UI THEME
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem; }
    .metric-box { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; border: 1px solid #e5e7eb; }
    .upload-box { background: white; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem 2rem; text-align: center; }
    .upload-box:hover { border-color: #667eea; background: #f9fafb; }
    .result-healthy { background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
    .result-warning { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
    .result-danger { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="main-header"><h1 style="margin:0;">🌿 AgriLeaf Analytics</h1><p style="margin:0.5rem 0 0 0; opacity:0.9;">AI-Powered Plant Pathology Detection</p></div>', unsafe_allow_html=True)

# METRICS
c1, c2, c3, c4 = st.columns(4)
c1.markdown('<div class="metric-box"><h3 style="color:#667eea; margin:0;">CNN</h3><p style="margin:0; color:#6b7280; font-size:0.9rem;">Architecture</p></div>', unsafe_allow_html=True)
c2.markdown('<div class="metric-box"><h3 style="color:#667eea; margin:0;">97%</h3><p style="margin:0; color:#6b7280; font-size:0.9rem;">Accuracy</p></div>', unsafe_allow_html=True)
c3.markdown('<div class="metric-box"><h3 style="color:#667eea; margin:0;">3</h3><p style="margin:0; color:#6b7280; font-size:0.9rem;">Classes</p></div>', unsafe_allow_html=True)
c4.markdown('<div class="metric-box"><h3 style="color:#667eea; margin:0;">256²</h3><p style="margin:0; color:#6b7280; font-size:0.9rem;">Input Size</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# TWO COLUMNS
left, right = st.columns(2)

with left:
    st.subheader("📤 Upload Sample")
    
    uploaded_file = st.file_uploader("Select leaf image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file:
        st.image(uploaded_file, use_column_width=True, caption="Input Preview")
        
        # FIX: Check session state to avoid rerun() causing loops
        if st.button("🔍 Analyze Sample", type="primary"):
            with st.spinner("Processing..."):
                label, confidence, all_probs = predict_disease(uploaded_file)
            
            # Store in session state (no rerun!)
            st.session_state['last_result'] = {
                'label': label, 
                'confidence': confidence, 
                'probs': all_probs,
                'img': uploaded_file
            }
    else:
        st.markdown('<div class="upload-box"><h4 style="color:#374151;">Drop leaf image here</h4><p style="color:#6b7280; font-size:0.9rem;">PNG, JPG supported</p></div>', unsafe_allow_html=True)

with right:
    st.subheader("📊 Diagnostic Results")
    
    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        label, conf = res['label'], res['confidence']
        
        # Display result box based on type
        if label == "Healthy":
            st.markdown(f'<div class="result-healthy"><h2 style="margin:0;">✓ {label}</h2><p style="opacity:0.9; margin:0.5rem 0 0 0;">Confidence: {conf*100:.1f}%</p></div>', unsafe_allow_html=True)
        elif label == "Powdery Mildew":
            st.markdown(f'<div class="result-warning"><h2 style="margin:0;">⚠ {label}</h2><p style="opacity:0.9; margin:0.5rem 0 0 0;">Confidence: {conf*100:.1f}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-danger"><h2 style="margin:0;">🦠 {label}</h2><p style="opacity:0.9; margin:0.5rem 0 0 0;">Confidence: {conf*100:.1f}%</p></div>', unsafe_allow_html=True)
        
        # Probabilities
        st.markdown("### Probability Distribution")
        for cls, prob in res['probs'].items():
            st.progress(prob, text=f"{cls}: {prob*100:.1f}%")
        
        # Recommendations
        st.markdown("### Recommendations")
        if label == "Healthy":
            st.success("No disease detected. Continue standard maintenance.")
        elif label == "Powdery Mildew":
            st.warning("Apply sulfur-based fungicides. Remove infected leaves.")
        else:
            st.error("Apply triazole fungicides immediately. Check surrounding plants.")
    else:
        st.info("Upload image and click Analyze to see results")

st.markdown("---")
st.caption("Team 4-KIET II | 2025 | TensorFlow Backend")