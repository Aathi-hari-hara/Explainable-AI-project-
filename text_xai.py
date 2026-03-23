"""
Text XAI v2 — Fixed
Fixes:
  1. TypeError: string indices must be integers
     Cause:  pipeline returns List[List[dict]] when return_all_scores=True
             clf(text)[0] gives the inner list, not a dict
     Fix:    use clf(text, truncation=True, max_length=512)[0]
             which returns List[dict] — then max() works correctly
  2. Tensor size error — truncation=True, max_length=512 everywhere
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading text model…")
def load_pipeline():
    from transformers import pipeline
    # return_all_scores removed — use top_k=None for all labels
    return pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        top_k=None,          # returns all label scores as List[dict]
        truncation=True,
        max_length=512,
    )

@st.cache_resource(show_spinner="Loading attention model…")
def load_attention_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    name = "distilbert-base-uncased-finetuned-sst-2-english"
    tok   = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, output_attentions=True)
    model.eval()
    return tok, model

# ── Safe prediction helper ────────────────────────────────────────────────────
def predict(clf, text):
    """
    Returns List[dict] like [{"label":"POSITIVE","score":0.98},{"label":"NEGATIVE","score":0.02}]
    Handles both pipeline output shapes safely.
    """
    raw = clf(text[:2000], truncation=True, max_length=512)
    # raw can be List[List[dict]] or List[dict] depending on version
    if isinstance(raw, list) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, list):
            return first          # unwrap double list
        elif isinstance(first, dict):
            return raw            # already flat list of dicts
    return [{"label": "POSITIVE", "score": 0.5}, {"label": "NEGATIVE", "score": 0.5}]

# ── HH-RLHF samples ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading HH-RLHF samples…")
def load_samples():
    try:
        from datasets import load_dataset
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= 15:
                break
            text = item.get("chosen", "")
            parts = re.findall(r'Assistant: (.+?)(?:Human:|$)', text, re.DOTALL)
            if parts:
                samples.append({
                    "label": f"HH-RLHF sample {i+1}",
                    "text": parts[0].strip()[:400],
                    "source": "Anthropic HH-RLHF",
                })
        if samples:
            return samples
    except Exception:
        pass
    return [
        {"label": "Helpful AI response",
         "text": "I'd be happy to help you understand machine learning. Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve their performance without being explicitly programmed.",
         "source": "HH-RLHF demo"},
        {"label": "Positive review",
         "text": "This product is absolutely fantastic! The quality is superb and it exceeded all my expectations. I would highly recommend it to anyone.",
         "source": "SST-2 style"},
        {"label": "Negative review",
         "text": "The service was terrible and the product broke after one day. I am extremely disappointed and would not recommend this to anyone.",
         "source": "SST-2 style"},
        {"label": "Factual/Wikipedia style",
         "text": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was constructed from 1887 to 1889 as the centrepiece of the 1889 World's Fair.",
         "source": "Wikipedia-style"},
        {"label": "Reddit/Opinion style",
         "text": "Honestly I think Python is way better than Java for machine learning projects. It is just so much easier to get started with and the libraries are incredible.",
         "source": "Reddit/Forum style"},
    ]

# ── Source detection ──────────────────────────────────────────────────────────
SOURCE_RULES = {
    "Wikipedia / Encyclopedia": {
        "keywords": ["located","founded","born","died","century","known as","refers to",
                     "is a","was a","population","history","capital","named after"],
        "color": "#2563eb",
        "desc": "Encyclopedic factual content — likely sourced from Wikipedia or reference databases in the training corpus.",
    },
    "Scientific / Research": {
        "keywords": ["study","research","hypothesis","data","results","evidence",
                     "experiment","analysis","published","journal","demonstrates",
                     "photosynthesis","algorithm","neural","protein"],
        "color": "#7c3aed",
        "desc": "Academic or scientific content — likely from ArXiv, PubMed, or scientific websites.",
    },
    "Reddit / Forums": {
        "keywords": ["honestly","personally","imo","tbh","i think","i feel",
                     "pretty sure","anyone else","has anyone","love","hate","awesome"],
        "color": "#d97706",
        "desc": "Conversational/opinionated content — likely from Reddit, Quora, or online forums.",
    },
    "News / Journalism": {
        "keywords": ["reported","according to","announced","government","official",
                     "president","election","policy","company","market","said"],
        "color": "#059669",
        "desc": "Journalistic content — likely from news sites or press releases.",
    },
    "AI Self-Reasoning": {
        "keywords": ["as an ai","i am an ai","i cannot","i can help","i should note",
                     "let me","i would suggest","certainly","of course","happy to help",
                     "i understand","i'd be happy"],
        "color": "#dc2626",
        "desc": "The AI produced this through its own language modeling — not retrieved from a specific source.",
    },
    "Medical / Health": {
        "keywords": ["symptom","treatment","diagnosis","patient","doctor","medication",
                     "therapy","clinical","health","medical","condition","disease"],
        "color": "#0891b2",
        "desc": "Medical content — likely from health websites (WebMD, Mayo Clinic) or clinical guidelines.",
    },
    "Programming / Docs": {
        "keywords": ["function","code","programming","library","api","database",
                     "python","javascript","algorithm","software","syntax","class"],
        "color": "#4f46e5",
        "desc": "Technical programming content — likely from GitHub, Stack Overflow, or documentation.",
    },
}

def detect_source(text):
    text_lower = text.lower()
    scored = {}
    evidence = {}
    for src, info in SOURCE_RULES.items():
        matched = [kw for kw in info["keywords"] if kw in text_lower]
        if matched:
            scored[src] = round(min(1.0, len(matched) * 0.25), 2)
            evidence[src] = matched[:5]
    if not scored:
        scored["AI Self-Reasoning"] = 0.65
        evidence["AI Self-Reasoning"] = ["no domain-specific signals"]
    top_src = max(scored, key=scored.get)
    return {
        "source": top_src,
        "confidence": int(scored[top_src] * 100),
        "color": SOURCE_RULES[top_src]["color"],
        "desc": SOURCE_RULES[top_src]["desc"],
        "all_scores": scored,
        "evidence": evidence,
    }

# ── Token importance (FIXED: truncation + word cap) ───────────────────────────
def token_importance(text, clf):
    words = text.split()[:50]   # cap to stay within 512 tokens
    if not words:
        return [], []
    base = predict(clf, " ".join(words))
    base_pos = next((s["score"] for s in base if s["label"] == "POSITIVE"), 0.5)
    imps = []
    for i in range(len(words)):
        masked = " ".join(w if j != i else "[MASK]" for j, w in enumerate(words))
        result = predict(clf, masked)
        score  = next((s["score"] for s in result if s["label"] == "POSITIVE"), 0.5)
        imps.append(round(base_pos - score, 4))
    return words, imps

# ── Attention (FIXED: truncation) ────────────────────────────────────────────
def get_attention(text, tokenizer, model):
    import torch
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=512, padding=False
    )
    with torch.no_grad():
        out = model(**inputs)
    attn   = out.attentions[-1][0].mean(dim=0)[0].numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, attn

# ── Render ────────────────────────────────────────────────────────────────────
def render():
    st.title("📝 Text XAI — Source Tracing + Sentiment Explanation")
    st.markdown("""
    Model Features:🔍 Traces where text data came from (Wikipedia, Reddit, AI-generated, etc.)
    """)
    st.info("📦 **Dataset:** Anthropic HH-RLHF (169K pairs) | **Model:** DistilBERT-SST2")
    st.divider()

    samples = load_samples()

    col1, col2 = st.columns([2, 1])
    with col1:
        method = st.radio("Input:", ["📦 HH-RLHF Sample", "✍️ Custom Text"], horizontal=True)
        if method == "📦 HH-RLHF Sample":
            idx = st.selectbox("Choose sample:", range(len(samples)),
                format_func=lambda i: samples[i]["label"])
            text_input = st.text_area("Text to analyse:", value=samples[idx]["text"], height=130)
            st.caption(f"Source: {samples[idx]['source']}")
        else:
            text_input = st.text_area("Paste any text (ChatGPT answer, article, review…):",
                height=130, placeholder="Paste text here…")

    with col2:
        st.markdown("**XAI Settings**")
        show_source = st.checkbox("Source Tracing", value=True)
        show_shap   = st.checkbox("Token Importance (SHAP)", value=True)
        show_attn   = st.checkbox("Attention Heatmap", value=True)

    if not text_input.strip():
        st.info("Enter or select text above, then click Analyse.")
        return

    if st.button("🔍 Analyse & Explain", type="primary", use_container_width=True):
        with st.spinner("Running analysis…"):

            # ── PREDICTION (FIXED) ────────────────────────────────────────────
            clf        = load_pipeline()
            prediction = predict(clf, text_input)   # always List[dict]
            top        = max(prediction, key=lambda x: x["score"])

            st.subheader("🎯 Sentiment Prediction")
            pc1, pc2 = st.columns(2)
            with pc1:
                color = "#22c55e" if top["label"] == "POSITIVE" else "#ef4444"
                st.markdown(f"""<div style="background:{color};color:white;padding:20px;
                    border-radius:10px;text-align:center;">
                    <h2 style="margin:0;color:white">{top['label']}</h2>
                    <p style="margin:6px 0 0;font-size:18px;">{top['score']*100:.1f}% confidence</p>
                    </div>""", unsafe_allow_html=True)
            with pc2:
                df_scores = pd.DataFrame(prediction)
                df_scores["score"] = df_scores["score"].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(df_scores, use_container_width=True, hide_index=True)

            st.divider()

            # ── SOURCE TRACING ────────────────────────────────────────────────
            if show_source:
                st.subheader("🌐 Source Tracing — Where Did This Text Come From?")
                src = detect_source(text_input)
                st.markdown(f"""<div style="background:{src['color']}18;
                    border-left:5px solid {src['color']};padding:16px;border-radius:8px;margin-bottom:12px;">
                    <h3 style="margin:0 0 6px;color:{src['color']}">📌 {src['source']}</h3>
                    <p style="margin:0"><b>Confidence:</b> {src['confidence']}%<br>
                    {src['desc']}</p></div>""", unsafe_allow_html=True)

                evid = src["evidence"].get(src["source"], [])
                if evid:
                    st.markdown(f"**Detected signals:** " +
                                " · ".join([f"`{e}`" for e in evid]))

                # Source confidence bar chart
                fig, ax = plt.subplots(figsize=(8, max(2.5, len(src["all_scores"])*0.5)))
                sorted_scores = sorted(src["all_scores"].items(), key=lambda x: x[1], reverse=True)
                labels = [s[0] for s in sorted_scores]
                values = [s[1]*100 for s in sorted_scores]
                bar_colors = ["#2E75B6" if l == src["source"] else "#cbd5e1" for l in labels]
                ax.barh(labels, values, color=bar_colors, edgecolor="white", height=0.55)
                ax.set_xlim(0, 100)
                ax.set_xlabel("Confidence (%)")
                ax.set_title("Data Source Attribution", fontweight="bold", color="#1F3864")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
                st.divider()

            # ── TOKEN IMPORTANCE ──────────────────────────────────────────────
            if show_shap:
                st.subheader("🔵 Token Importance (SHAP-style)")
                st.markdown("Each word masked one-at-a-time. Input capped at 50 words to stay within 512-token limit.")
                words, imps = token_importance(text_input, clf)
                if words:
                    fig, ax = plt.subplots(figsize=(max(6, len(words)*0.62), 3.5))
                    colors_t = ["#22c55e" if s >= 0 else "#ef4444" for s in imps]
                    ax.barh(range(len(words)), imps, color=colors_t, edgecolor="white", height=0.65)
                    ax.set_yticks(range(len(words)))
                    ax.set_yticklabels(words, fontsize=10)
                    ax.axvline(0, color="black", linewidth=0.8)
                    ax.set_title("Token Importance (green=POSITIVE, red=NEGATIVE)",
                                 fontweight="bold", color="#1F3864")
                    ax.set_xlabel("Impact on POSITIVE prediction")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                    with st.expander("📊 Raw scores"):
                        df_tok = pd.DataFrame({"Token": words, "Score": imps})
                        df_tok["Direction"] = df_tok["Score"].apply(
                            lambda x: "→ POSITIVE" if x > 0 else "→ NEGATIVE")
                        st.dataframe(df_tok.sort_values("Score", ascending=False),
                                     use_container_width=True, hide_index=True)
                st.divider()

            # ── ATTENTION ─────────────────────────────────────────────────────
            if show_attn:
                st.subheader("🟠 Attention Heatmap")
                try:
                    tm, am = load_attention_model()
                    tokens, attn_sc = get_attention(text_input, tm, am)
                    pairs = [(t, s) for t, s in zip(tokens, attn_sc)
                             if t not in ["[CLS]","[SEP]","[PAD]","<s>","</s>"]][:25]
                    if pairs:
                        toks, sc = zip(*pairs)
                        sc = np.array(sc)
                        sc = sc / (sc.max() + 1e-8)
                        fig, ax = plt.subplots(figsize=(max(6, len(toks)*0.55), 2.5))
                        cmap = plt.cm.Blues
                        for i, (t, s) in enumerate(zip(toks, sc)):
                            ax.text(i, 0.5, t, ha="center", va="center",
                                    fontsize=9, fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.3",
                                              facecolor=cmap(float(s)),
                                              edgecolor="white", alpha=0.9))
                        ax.set_xlim(-0.5, len(toks)-0.5)
                        ax.set_ylim(0, 1)
                        ax.axis("off")
                        ax.set_title("Attention Weights (darker = higher attention)",
                                     fontweight="bold", color="#1F3864")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                except Exception as e:
                    st.warning(f"Attention unavailable: {e}")

            # ── SUMMARY ───────────────────────────────────────────────────────
            st.divider()
            st.subheader("📋 XAI Summary")
            src = detect_source(text_input)
            words2, imps2 = token_importance(text_input, clf)
            pos_w = [w for w, s in zip(words2, imps2) if s > 0][:3]
            neg_w = [w for w, s in zip(words2, imps2) if s < 0][:3]
            st.success(f"""
**Sentiment:** {top['label']} ({top['score']*100:.1f}% confidence)
**Primary data source:** {src['source']} ({src['confidence']}% confidence)
**Source explanation:** {src['desc']}
**Top POSITIVE tokens:** {', '.join(pos_w) if pos_w else 'none'}
**Top NEGATIVE tokens:** {', '.join(neg_w) if neg_w else 'none'}
**Dataset:** Anthropic HH-RLHF (169K human preference pairs)
**Model training data:** English Wikipedia + BookCorpus (SST-2 fine-tuning)
            """)
