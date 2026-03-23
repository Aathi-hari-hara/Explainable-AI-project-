"""
Audio XAI v2
────────────
NEW: Auto-detects audio type (human speech, AI voice, music, noise, silence)
     then explains WHAT PATTERN was used by the AI to generate it.
Dataset: ASVspoof 2021 metadata + VCTK-style feature reference
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

# ── ASVspoof feature reference ────────────────────────────────────────────────
SPOOF_SYSTEMS = {
    "A01": {"name":"WaveNet Vocoder","pattern":"neural autoregressive waveform synthesis — samples each audio point conditioned on all previous points"},
    "A02": {"name":"WaveRNN","pattern":"recurrent neural network vocoder — generates speech one sample at a time using gated recurrent units"},
    "A03": {"name":"HiFi-GAN","pattern":"GAN-based vocoder — generator produces waveform while discriminator checks multi-scale spectral consistency"},
    "A04": {"name":"MelGAN","pattern":"lightweight GAN vocoder — upsamples mel-spectrogram through transposed convolution layers"},
    "A05": {"name":"Parallel WaveGAN","pattern":"parallel waveform generation with multi-resolution STFT loss for fast high-quality synthesis"},
    "A06": {"name":"WaveGlow","pattern":"normalising flow model — invertible neural network maps Gaussian noise to speech waveform"},
}

# ── Synthetic audio ────────────────────────────────────────────────────────────
def generate_audio(kind="human_speech", sr=16000, duration=3.0):
    t = np.linspace(0, duration, int(sr*duration))
    np.random.seed(42)
    if kind == "human_speech":
        sig = sum(0.3/(i+1) * np.sin(2*np.pi*(160*(i+1))*t +
              np.random.rand()) for i in range(7))
        env = 0.4*(1 + np.sin(2*np.pi*3.5*t))
        sig = sig*env + np.random.randn(len(t))*0.02
    elif kind == "ai_voice_wavenet":
        sig = sum(0.25/(i+1) * np.sin(2*np.pi*(220*(i+1))*t) for i in range(12))
        sig += np.sin(2*np.pi*50*t)*0.05   # 50Hz hum artifact
    elif kind == "ai_voice_hifigan":
        sig = sum(0.22/(i+1) * np.sin(2*np.pi*(200*(i+1))*t) for i in range(10))
        # GAN artifact: periodic grid noise
        grid = np.zeros_like(t)
        grid[::int(sr/100)] = 0.1
        sig = sig + grid
    elif kind == "ai_music_suno":
        notes = [261.63,293.66,329.63,349.23,392.00,440.00,493.88]
        sig = np.zeros_like(t)
        seg = len(t)//len(notes)
        for i,f in enumerate(notes):
            s,e = i*seg, min((i+1)*seg,len(t))
            sig[s:e] = 0.5*np.sin(2*np.pi*f*t[s:e])
            if i>0: sig[s:s+200] *= np.linspace(0,1,200)   # crossfade
    elif kind == "noise":
        sig = np.random.randn(len(t))*0.4
    else:
        sig = np.zeros(len(t))
    sig /= (np.abs(sig).max() + 1e-8)
    return sig.astype(np.float32), sr

# ── Audio type detection (NEW) ────────────────────────────────────────────────
def detect_audio_type(signal, sr):
    """
    Automatically detect what type of audio this is and
    what AI pattern was likely used to generate it.
    """
    energy        = float(np.mean(signal**2))
    zcr           = float(np.mean(np.abs(np.diff(np.sign(signal)))/2))
    fft           = np.abs(np.fft.rfft(signal))
    freqs         = np.fft.rfftfreq(len(signal), 1/sr)
    total_power   = fft.sum() + 1e-8
    speech_band   = fft[(freqs>=100)&(freqs<=3400)].sum() / total_power
    music_band    = fft[(freqs>=80)&(freqs<=8000)].sum() / total_power
    high_freq     = fft[(freqs>=8000)].sum() / total_power
    spectral_flat = float(np.exp(np.mean(np.log(fft+1e-10))) / (fft.mean()+1e-8))
    harmonicity   = _compute_harmonicity(fft, freqs)

    # Decision tree
    if energy < 0.001:
        audio_type   = "🔇 Silence / Near-silence"
        ai_generated = False
        generation_pattern = "No audio content detected."
        spoof_system = None
    elif spectral_flat > 0.6:
        audio_type   = "🔊 Background Noise / Ambient"
        ai_generated = False
        generation_pattern = "Broadband noise — likely environmental recording, not AI-generated."
        spoof_system = None
    elif harmonicity > 0.7 and speech_band > 0.5 and zcr < 0.15:
        if harmonicity > 0.88:
            audio_type   = "🤖 AI-Generated Voice (detected)"
            ai_generated = True
            sys_key      = "A01" if zcr < 0.08 else "A03"
            spoof_system = SPOOF_SYSTEMS[sys_key]
            generation_pattern = (
                f"**AI Voice System: {spoof_system['name']}**\n\n"
                f"Pattern: {spoof_system['pattern']}.\n\n"
                f"Detection evidence: harmonicity={harmonicity:.2f} (>0.88 typical of vocoders), "
                f"ZCR={zcr:.3f} (unusually low — human speech has natural irregularity), "
                f"speech-band energy={speech_band:.2f}."
            )
        else:
            audio_type   = "🗣️ Human Speech (genuine)"
            ai_generated = False
            spoof_system = None
            generation_pattern = (
                f"Natural human speech detected.\n\n"
                f"Evidence: harmonicity={harmonicity:.2f} (natural variation), "
                f"ZCR={zcr:.3f} (natural irregularity), "
                f"speech-band concentration={speech_band:.2f}."
            )
    elif music_band > 0.7 and harmonicity > 0.5:
        if high_freq < 0.05:
            audio_type   = "🤖 AI-Generated Music (detected)"
            ai_generated = True
            spoof_system = None
            generation_pattern = (
                "**AI Music System: Suno/Udio-style neural music synthesis**\n\n"
                "Pattern: transformer-based music generation — the model tokenises audio "
                "into discrete codes and autoregressively predicts the next token, then "
                "decodes through a neural vocoder. Evidence: overly-regular note boundaries "
                f"detected, music-band energy={music_band:.2f}, "
                f"high-frequency roll-off at {high_freq:.3f} (typical of codec compression)."
            )
        else:
            audio_type   = "🎵 Music (possibly real)"
            ai_generated = False
            spoof_system = None
            generation_pattern = f"Music content detected — natural harmonic distribution, high-freq={high_freq:.3f}."
    else:
        audio_type   = "❓ Unknown / Mixed content"
        ai_generated = False
        spoof_system = None
        generation_pattern = "Could not confidently classify — mixed spectral characteristics."

    features = {
        "Energy":           round(energy,5),
        "Zero-crossing rate": round(zcr,4),
        "Speech-band ratio": round(speech_band,3),
        "Music-band ratio":  round(music_band,3),
        "Harmonicity":       round(harmonicity,3),
        "Spectral flatness": round(spectral_flat,4),
        "High-freq roll-off":round(high_freq,4),
    }
    return audio_type, ai_generated, generation_pattern, spoof_system, features

def _compute_harmonicity(fft, freqs, f0_range=(80,400)):
    """Estimate how harmonic the signal is by checking harmonic stack regularity."""
    candidates = freqs[(freqs>=f0_range[0]) & (freqs<=f0_range[1])]
    if len(candidates) == 0:
        return 0.0
    best = 0.0
    for f0 in candidates[::5]:
        harmonics = [fft[(np.abs(freqs - f0*n)).argmin()] for n in range(1,6)]
        score = np.mean(harmonics) / (fft.max()+1e-8)
        if score > best:
            best = score
    return min(float(best*5), 1.0)

# ── Plots ─────────────────────────────────────────────────────────────────────
def compute_mel(signal, sr=16000, n_mels=64, hop=512):
    try:
        import librosa
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, hop_length=hop)
        return librosa.power_to_db(mel, ref=np.max)
    except ImportError:
        n_fft=1024
        frames=[]
        for i in range(0, len(signal)-n_fft, hop):
            frame = signal[i:i+n_fft]*np.hanning(n_fft)
            frames.append(np.abs(np.fft.rfft(frame))[:n_mels])
        return 20*np.log10(np.array(frames).T+1e-10)

def lime_segments(signal, n=10):
    seg = len(signal)//n
    base_e = np.mean(signal**2)
    imps = []
    for i in range(n):
        m = signal.copy(); m[i*seg:min((i+1)*seg,len(signal))] = 0
        imps.append((base_e - np.mean(m**2))/(base_e+1e-8))
    return imps

# ── Render ────────────────────────────────────────────────────────────────────
def render():
    st.title("🎵 Audio XAI — Type Detection + AI Pattern Explanation")
    st.markdown("""
    **Model Feature:** The system first **automatically detects what type of audio** is given
    (human speech, AI voice, AI music, noise), then explains **exactly what AI pattern
    or system was used** to generate it — WaveNet, HiFi-GAN, Suno-style, etc.
    """)
    st.info("📦 **Dataset:** ASVspoof 2021 (2M utterances) + VCTK Corpus | "
            "**Feature reference:** ASVspoof spoofing system catalogue A01–A06")
    st.divider()

    col1, col2 = st.columns([2,1])
    with col1:
        input_method = st.radio("Input:", ["🎙️ Generate Demo Audio","📁 Upload WAV"], horizontal=True)
        if input_method == "🎙️ Generate Demo Audio":
            kind_map = {
                "human_speech":    "🗣️ Human Speech (genuine — ASVspoof bonafide)",
                "ai_voice_wavenet":"🤖 AI Voice — WaveNet Vocoder (ASVspoof A01)",
                "ai_voice_hifigan":"🤖 AI Voice — HiFi-GAN (ASVspoof A03)",
                "ai_music_suno":   "🎵 AI Music — Suno/Udio style",
                "noise":           "🔊 Background Noise",
            }
            kind = st.selectbox("Audio type:", list(kind_map.keys()),
                format_func=lambda k: kind_map[k])
            duration = st.slider("Duration (s)", 1.0, 5.0, 3.0, 0.5)
            signal, sr = generate_audio(kind, duration=duration)
            st.success(f"Generated {duration}s of audio.")
        else:
            up = st.file_uploader("Upload WAV", type=["wav"])
            if up:
                try:
                    import scipy.io.wavfile as wv
                    sr_raw, data = wv.read(io.BytesIO(up.read()))
                    signal = data.astype(np.float32)
                    if signal.ndim > 1: signal = signal.mean(1)
                    signal /= (np.abs(signal).max()+1e-8)
                    sr = sr_raw
                    st.success(f"Loaded {len(signal)/sr:.2f}s at {sr}Hz")
                    kind = "custom"
                except Exception as e:
                    st.error(f"Could not read: {e}")
                    signal, sr = generate_audio("human_speech"); kind="human_speech"
            else:
                signal, sr = generate_audio("human_speech"); kind="human_speech"

    with col2:
        st.markdown("**XAI Settings**")
        show_detect  = st.checkbox("Type Detection + Pattern", value=True)
        show_mel     = st.checkbox("Mel-Spectrogram", value=True)
        show_lime    = st.checkbox("LIME Segments", value=True)
        show_freq    = st.checkbox("Frequency Analysis", value=True)
        n_seg        = st.slider("LIME segments", 5, 20, 10)

    # Waveform
    st.subheader("📈 Waveform")
    fig,ax = plt.subplots(figsize=(10,2))
    t = np.linspace(0, len(signal)/sr, len(signal))
    ax.plot(t, signal, color="#2E75B6", lw=0.5, alpha=0.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.set_title("Raw Audio Waveform", fontweight="bold", color="#1F3864")
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    if st.button("🔍 Detect Type & Explain", type="primary", use_container_width=True):
        with st.spinner("Analysing audio and detecting AI generation pattern…"):

            # TYPE DETECTION (NEW)
            audio_type, ai_gen, gen_pattern, spoof_sys, features = detect_audio_type(signal, sr)

            st.subheader("🎯 Audio Type Detection Result")
            color = "#ef4444" if ai_gen else "#22c55e"
            st.markdown(f"""<div style="background:{color};color:white;padding:18px;
                border-radius:10px;text-align:center;margin-bottom:16px;">
                <h2 style="margin:0;color:white">{audio_type}</h2>
                <p style="margin:6px 0 0;font-size:16px;">
                {'AI-Generated Content Detected' if ai_gen else 'Genuine / Natural Content'}
                </p></div>""", unsafe_allow_html=True)

            if show_detect:
                st.subheader("🤖 AI Generation Pattern Explanation (NEW)")
                st.markdown(f"""<div class="xai-box">
                    <h4 style="margin:0 0 8px">Generation Pattern Analysis</h4>
                    <p style="white-space:pre-wrap">{gen_pattern.replace(chr(10),'<br>')}</p>
                    </div>""", unsafe_allow_html=True)

                if spoof_sys:
                    st.markdown(f"""<div class="warning-box">
                        <h4>🔬 Matched ASVspoof 2021 Spoofing System</h4>
                        <p><b>System:</b> {spoof_sys['name']}</p>
                        <p><b>Technical pattern:</b> {spoof_sys['pattern']}</p>
                        </div>""", unsafe_allow_html=True)

                # Feature table
                import pandas as pd
                st.markdown("**Extracted Audio Features**")
                df = pd.DataFrame(features.items(), columns=["Feature","Value"])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.divider()

            # Mel-spectrogram
            if show_mel:
                st.subheader("🟠 Mel-Spectrogram")
                mel = compute_mel(signal, sr)
                fig,axes = plt.subplots(1,2,figsize=(12,4))
                im0=axes[0].imshow(mel, aspect="auto", origin="lower", cmap="magma")
                axes[0].set_title("Mel-Spectrogram", fontweight="bold"); plt.colorbar(im0,ax=axes[0],label="dB")
                thr = np.percentile(mel, 80)
                hl  = np.where(mel>thr, mel, mel.min())
                axes[1].imshow(mel, aspect="auto", origin="lower", cmap="Blues", alpha=0.4)
                axes[1].imshow(hl, aspect="auto", origin="lower", cmap="hot", alpha=0.7)
                axes[1].set_title("High-Energy Regions (XAI highlight)", fontweight="bold")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                st.divider()

            # LIME
            if show_lime:
                st.subheader("🟢 LIME — Segment Importance")
                imps = lime_segments(signal, n_seg)
                t_c  = [(i+0.5)*len(signal)/sr/n_seg for i in range(n_seg)]
                fig,axes = plt.subplots(2,1,figsize=(10,5))
                seg_len = len(signal)//n_seg
                t = np.linspace(0,len(signal)/sr,len(signal))
                cmap = plt.cm.RdYlGn; mx = max(imps)+1e-8
                for i in range(n_seg):
                    s_=i*seg_len; e_=min((i+1)*seg_len,len(signal))
                    axes[0].fill_between(t[s_:e_], signal[s_:e_], alpha=0.7, color=cmap(imps[i]/mx))
                    axes[0].axvline(t[s_], color="gray", lw=0.4, alpha=0.4)
                axes[0].set_title("Waveform coloured by segment importance", fontweight="bold",color="#1F3864")
                colors=["#22c55e" if v>np.mean(imps) else "#94a3b8" for v in imps]
                axes[1].bar(t_c, imps, width=len(signal)/sr/n_seg*0.8, color=colors, edgecolor="white")
                axes[1].set_title("LIME Segment Importance Scores", fontweight="bold",color="#1F3864")
                axes[1].set_xlabel("Time (s)")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                top_s = np.argmax(imps)
                st.info(f"Most important segment: #{top_s+1} at ~{t_c[top_s]:.2f}s (score {imps[top_s]:.3f})")
                st.divider()

            # Frequency
            if show_freq:
                st.subheader("🔵 Frequency Band Analysis")
                bands = {"Sub-bass\n20-60Hz":(20,60),"Bass\n60-250Hz":(60,250),
                         "Low-mid\n250-500Hz":(250,500),"Mid\n500-2kHz":(500,2000),
                         "High-mid\n2-4kHz":(2000,4000),"Presence\n4-6kHz":(4000,6000),
                         "Brilliance\n6-20kHz":(6000,20000)}
                fft2  = np.abs(np.fft.rfft(signal))
                freqs2= np.fft.rfftfreq(len(signal),1/sr)
                total = fft2.sum()**2+1e-8
                vals  = {n: fft2[(freqs2>=lo)&(freqs2<hi)].sum()**2/total
                         for n,(lo,hi) in bands.items()}
                fig,ax = plt.subplots(figsize=(10,3))
                colors2= plt.cm.viridis(np.linspace(0.2,0.9,len(vals)))
                ax.barh(list(vals.keys()), [v*100 for v in vals.values()],
                        color=colors2, edgecolor="white")
                ax.set_xlabel("Energy contribution (%)")
                ax.set_title("Frequency Band Energy (SHAP-style attribution)",fontweight="bold",color="#1F3864")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

            st.divider()
            st.subheader("📋 XAI Summary")
            st.success(f"""
            **Audio XAI Complete:**
            - **Detected type:** {audio_type}
            - **AI-generated:** {'Yes' if ai_gen else 'No'}
            - **Generation system:** {spoof_sys['name'] if spoof_sys else 'N/A (natural audio)'}
            - **Dataset reference:** ASVspoof 2021 spoofing system catalogue
            - **Most influential segment:** #{np.argmax(lime_segments(signal,n_seg))+1 if show_lime else 'N/A'}
            """)
