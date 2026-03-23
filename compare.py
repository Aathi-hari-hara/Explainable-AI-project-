import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def render():
    st.title("📊 Cross-Domain XAI Comparison ")
    st.divider()

    st.subheader("🆕 v2 Feature Summary by Domain")
    data = {
        "Domain":   ["📝 Text","🖼️ Image","🎵 Audio","💻 Code","🎬 Video"],
        " New Feature": [
            "Data source tracing (Wikipedia / Reddit / AI-generated)",
            "Natural-language generation pattern summary",
            "Auto audio type detection + AI system identification",
            "Language auto-detect + logic/approach explanation",
            "Prompt analysis → step-by-step generation explanation",
        ],
        "Real Dataset": [
            "Anthropic HH-RLHF (169K pairs)",
            "DiffusionDB (14M images)",
            "ASVspoof 2021 (2M utterances)",
            "CodeSearchNet (2M pairs)",
            "UCF-101 + FakeAVCeleb",
        ],
        "XAI Method": [
            "SHAP + Attention + Source Tracing",
            "GradCAM + Pattern Analysis",
            "LIME + Freq SHAP + Type Detection",
            "Token SHAP + Logic Explainer",
            "Temporal GradCAM + Prompt Parser",
        ],
        "Error Fixed": ["✅ Tensor size (512 truncation)","—","—","—","—"],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    st.divider()

    st.subheader("📈 XAI Method Performance (Simulated)")
    domains  = ["Text","Image","Audio","Code","Video"]
    methods  = ["SHAP","LIME","GradCAM","Attention"]
    scores   = {
        "SHAP":      [0.84,0.76,0.72,0.86,0.69],
        "LIME":      [0.79,0.73,0.75,0.71,0.66],
        "GradCAM":   [0.56,0.89,0.61,0.51,0.84],
        "Attention": [0.86,0.71,0.66,0.83,0.73],
    }
    fig,ax = plt.subplots(figsize=(10,4))
    x = np.arange(len(domains)); w=0.18
    colors=["#2E75B6","#22c55e","#f59e0b","#a855f7"]
    for i,(method,vals) in enumerate(scores.items()):
        ax.bar(x+i*w-1.5*w, vals, w, label=method, color=colors[i], edgecolor="white", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(domains)
    ax.set_ylim(0,1.0); ax.set_ylabel("Faithfulness Score")
    ax.set_title("XAI Faithfulness by Domain & Method",fontweight="bold",color="#1F3864")
    ax.legend(); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    st.divider()

