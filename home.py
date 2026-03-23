import streamlit as st

def render():
    st.title("🔍 XAI Explainable AI")
    st.markdown("""
    > Explains **why** AI models produce specific outputs across 5 domains.
    """)
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl) in zip([c1,c2,c3,c4,c5],[
        ("5","Domains"),("Real","Datasets"),("10+","XAI Methods"),("5","AI Models")
    ]):
        col.markdown(f"""<div style="background:linear-gradient(135deg,#1F3864,#2E75B6);
        color:white;padding:16px;border-radius:10px;text-align:center;">
        <h2 style="margin:0;color:white">{val}</h2><p style="margin:4px 0 0 0;font-size:13px">{lbl}</p>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Domains")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
**📝 Text XAI**
- Fixed tensor size error (truncation to 512 tokens)
- Traces where the text data came from (Wikipedia, AI-generated, News, etc.)
- Shows reasoning path of LLM answer

**🖼️ Image XAI**
- GradCAM saliency maps
- Text summary of what patterns were used to generate the image
- Identifies textures, shapes, styles, objects

**🎵 Audio XAI**
- Auto-detects audio type (speech, music, noise, AI-generated voice)
- Identifies AI generation patterns (TTS artifacts, GAN fingerprints)
- No manual type selection needed
        """)
    with col2:
        st.markdown("""
**💻 Code XAI**
- Auto-detects programming language (no user input needed)
- Explains WHY the AI chose that particular logic/approach
- Shows which method/algorithm was selected and why
- Explains the solution strategy

**🎬 Video XAI**
- Analyses prompt → how it was interpreted to generate video
- Shows which visual concepts were derived from the prompt
- Explains scene composition, motion, style decisions

**📊 Compare All**
- Cross-domain XAI comparison dashboard
- Dataset overview and metrics
        """)

    st.divider()
    st.subheader("🗄️ Real Datasets Used")
    import pandas as pd
    df = pd.DataFrame({
        "Domain":  ["Text","Text","Image","Image","Audio","Audio","Code","Code","Video","Video"],
        "Dataset": ["HH-RLHF","TruthfulQA","DiffusionDB","CIFAKE",
                    "ASVspoof 2021","VCTK Corpus","CodeSearchNet","HumanEval",
                    "FakeAVCeleb","UCF-101"],
        "Size":    ["169K","817","14M","120K","2M","44K","2M","164","19.5K","13.3K"],
        "Source":  ["Anthropic/HF","OpenAI","HuggingFace","Kaggle",
                    "asvspoof.org","Edinburgh","GitHub/HF","OpenAI","Research","UCF"],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.info("👈 Select a domain from the sidebar to start exploring.")
