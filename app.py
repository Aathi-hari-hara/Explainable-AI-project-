"""
Explainable AI for AI-Generated Content
Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="XAI Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.domain-card{background:white;border-radius:12px;padding:20px;margin:8px 0;
  border-left:5px solid #2E75B6;box-shadow:0 2px 8px rgba(0,0,0,0.08);}
.insight-box{background:#f0f7ff;border-left:4px solid #2E75B6;padding:14px;
  border-radius:6px;margin:10px 0;}
.warning-box{background:#fff8e1;border-left:4px solid #f59e0b;padding:14px;
  border-radius:6px;margin:10px 0;}
.success-box{background:#f0fff4;border-left:4px solid #22c55e;padding:14px;
  border-radius:6px;margin:10px 0;}
h1,h2,h3{color:#1F3864;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔍 XAI Dashboard ")
    st.markdown("Explainable AI ")
    st.divider()
    st.markdown("Please Select a Domain:")
    page = st.radio("Navigate", [
        "🏠 Home","📝 Text XAI","🖼️ Image XAI",
        "🎵 Audio XAI","💻 Code XAI","🎬 Video XAI","📊 Compare All"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**XAI Techniques**")
    st.markdown("""
- 🔵 SHAP — Feature attribution
- 🟢 LIME — Local explanations
- 🟣 GradCAM — Image saliency
- 🟠 Attention — Token weights
- 🔴 Integrated Gradients
    """)
    

if   page == "🏠 Home":        from modules import home;      home.render()
elif page == "📝 Text XAI":    from modules import text_xai;  text_xai.render()
elif page == "🖼️ Image XAI":   from modules import image_xai; image_xai.render()
elif page == "🎵 Audio XAI":   from modules import audio_xai; audio_xai.render()
elif page == "💻 Code XAI":    from modules import code_xai;  code_xai.render()
elif page == "🎬 Video XAI":   from modules import video_xai; video_xai.render()
elif page == "📊 Compare All": from modules import compare;   compare.render()
