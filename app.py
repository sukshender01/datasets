import os
import io
import time
import tempfile
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

import sacrebleu
from jiwer import wer, cer, Compose, RemovePunctuation, ToLowerCase, Strip, RemoveMultipleSpaces
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="JP→EN Translation Evaluator", layout="wide")
st.title("🎌 JP → EN Translation Evaluator")
st.caption("ASR (JP) → NMT (EN), with BLEU, WER, Local Agreement (LA), and Token Delay metrics + PDF report")


# -----------------------------
# Sidebar: inputs
# -----------------------------
st.sidebar.header("Dataset Inputs")
dataset_mode = st.sidebar.selectbox(
    "Input type",
    ["Common Voice TSV (validated.tsv)", "Matched CSV (matched_dataset.csv)"]
)

tsv_or_csv_path = st.sidebar.text_input(
    "Path to validated.tsv or matched_dataset.csv",
    value=""
)

clips_dir = st.sidebar.text_input(
    "Path to MP3 clips folder",
    value=""
)

max_samples = st.sidebar.number_input("Max samples to evaluate (0 = all)", min_value=0, value=50, step=10)
asr_model_size = st.sidebar.selectbox("Whisper model", ["small", "medium", "large-v3"])
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device detected: **{device.upper()}**")

run_btn = st.sidebar.button("▶️ Run Evaluation")


# -----------------------------
# Utility: load dataset
# -----------------------------
def load_dataset_table(path: str, mode: str, clips: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - audio_path (absolute path to mp3)
      - ref_ja (Japanese transcript if present; for CV it's usually English sentence; leave empty if absent)
      - ref_en (English reference translation; optional, used for metrics)
    For CV 'validated.tsv', we take columns: path, sentence.
    For 'matched_dataset.csv', expect columns: mp3_full_path or filename/path, sentence/[source_lang]/[target_lang].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not os.path.isdir(clips):
        raise NotADirectoryError(f"Clips folder not found: {clips}")

    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    df = df.copy()

    # Normalize columns
    if mode.startswith("Common Voice"):
        # CV validated.tsv has 'path' (filename) and 'sentence' (text; often EN for CV-en)
        if "path" not in df.columns:
            raise ValueError("validated.tsv must contain a 'path' column.")
        df["audio_path"] = df["path"].apply(lambda x: os.path.join(clips, x))
        # In JP datasets you'd have Japanese; for CV-en this will be English text (can serve as ref_en).
        if "sentence" in df.columns:
            df["ref_en"] = df["sentence"].astype(str)
        else:
            df["ref_en"] = ""
        df["ref_ja"] = ""  # unknown for CV-en
    else:
        # matched_dataset.csv variants
        # Try common column names
        if "mp3_full_path" in df.columns:
            df["audio_path"] = df["mp3_full_path"]
        elif "filename" in df.columns:
            # If stored relative ("clips/xxx.mp3"), join with clips dir
            df["audio_path"] = df["filename"].apply(
                lambda x: x if os.path.isabs(str(x)) else os.path.join(clips, str(x))
            )
        elif "path" in df.columns:
            df["audio_path"] = df["path"].apply(
                lambda x: x if os.path.isabs(str(x)) else os.path.join(clips, str(x))
            )
        else:
            raise ValueError("CSV must include one of: mp3_full_path, filename, or path.")

        # Reference columns (best effort):
        if "target_lang" in df.columns:
            df["ref_en"] = df["target_lang"].astype(str)
        elif "translation" in df.columns:
            df["ref_en"] = df["translation"].astype(str)
        elif "sentence" in df.columns:
            df["ref_en"] = df["sentence"].astype(str)
        else:
            df["ref_en"] = ""

        if "source_lang" in df.columns:
            df["ref_ja"] = df["source_lang"].astype(str)
        else:
            df["ref_ja"] = ""

    # Keep only rows with existing audio
    df["exists"] = df["audio_path"].apply(os.path.exists)
    df = df[df["exists"]].drop(columns=["exists"])
    return df[["audio_path", "ref_ja", "ref_en"]]


# -----------------------------
# Caching models
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_asr(model_size: str, dev: str):
    # Whisper model supports word timestamps with beam_search at some sizes
    compute_type = "float16" if dev == "cuda" else "int8"
    model = WhisperModel(model_size, device=dev, compute_type=compute_type)
    return model

@st.cache_resource(show_spinner=False)
def load_nmt():
    # Solid JP→EN baseline
    model_name = "Helsinki-NLP/opus-mt-ja-en"
    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mod.to(device)
    return tok, mod


# -----------------------------
# Metrics
# -----------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())

def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    # sacrebleu expects list of refs lists
    refs = [[normalize_text(r) for r in references]]
    hyps = [normalize_text(h) for h in hypotheses]
    return float(sacrebleu.corpus_bleu(hyps, refs).score)

def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    # Lowercase & basic normalization for fair WER
    tx = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
    refs = [tx(r) for r in references]
    hyps = [tx(h) for h in hypotheses]
    # jiwer.cer/wer expect a single pair; we average over samples
    vals = [wer(r, h) for r, h in zip(refs, hyps)]
    return float(np.mean(vals)) if vals else 0.0

def local_agreement(ref: str, hyp: str) -> float:
    """
    Simple Local Agreement (LA) proxy: ratio of tokens matching in aligned positions
    after basic whitespace tokenization. Uses length of the shorter sequence.
    """
    ref_toks = normalize_text(ref).split()
    hyp_toks = normalize_text(hyp).split()
    n = min(len(ref_toks), len(hyp_toks))
    if n == 0:
        return 0.0
    agree = sum(1 for i in range(n) if ref_toks[i] == hyp_toks[i])
    return agree / n

def compute_la(references: List[str], hypotheses: List[str]) -> float:
    vals = [local_agreement(r, h) for r, h in zip(references, hypotheses)]
    return float(np.mean(vals)) if vals else 0.0

def token_delay_proxy(audio_dur_sec: float, hyp_tokens: int) -> float:
    """
    A lightweight proxy for Average Token Delay (ATD):
    assume tokens are emitted uniformly over the audio.
    ATD = (audio_dur / hyp_tokens) in seconds.
    Lower is better (faster emissions). This is a proxy, not true streaming AL.
    """
    if hyp_tokens <= 0:
        return audio_dur_sec
    return audio_dur_sec / hyp_tokens


# -----------------------------
# Pipeline: ASR (JP) → NMT (EN)
# -----------------------------
def transcribe_japanese(asr_model: WhisperModel, audio_path: str) -> Tuple[str, float]:
    """
    Transcribe Japanese audio; returns transcript text and audio duration (s).
    """
    # Faster-Whisper returns segments with timestamps; enable vad/filtering defaults
    segments, info = asr_model.transcribe(
        audio_path,
        language="ja",
        beam_size=5,
        vad_filter=True,
        word_timestamps=False
    )
    text = "".join(seg.text for seg in segments).strip()
    dur = info.duration if hasattr(info, "duration") else 0.0
    return text, float(dur)

def translate_ja_en(tok, mod, text_ja: str, max_new_tokens: int = 128) -> str:
    if not text_ja:
        return ""
    inputs = tok([text_ja], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = mod.generate(**inputs, max_new_tokens=max_new_tokens)
    en = tok.batch_decode(out, skip_special_tokens=True)[0]
    return en.strip()


# -----------------------------
# Main run
# -----------------------------
if run_btn:
    try:
        df = load_dataset_table(tsv_or_csv_path, dataset_mode, clips_dir)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    if max_samples and max_samples > 0:
        df = df.iloc[:max_samples].copy()

    st.success(f"Loaded {len(df)} audio items.")
    st.dataframe(df.head(10))

    # Load models
    with st.spinner("Loading ASR & NMT models..."):
        asr = load_asr(asr_model_size, device)
        tok, nmt = load_nmt()

    results = []
    progress = st.progress(0)
    start_time = time.time()

    for idx, row in df.iterrows():
        audio_path = row["audio_path"]
        ref_ja = row.get("ref_ja", "")
        ref_en = row.get("ref_en", "")

        # ASR
        try:
            hyp_ja, dur = transcribe_japanese(asr, audio_path)
        except Exception as e:
            hyp_ja, dur = "", 0.0
            st.warning(f"ASR failed for {os.path.basename(audio_path)}: {e}")

        # NMT
        try:
            hyp_en = translate_ja_en(tok, nmt, hyp_ja)
        except Exception as e:
            hyp_en = ""
            st.warning(f"NMT failed for sample: {e}")

        # Metrics per-sample
        # For BLEU/WER we need references; if ref_en empty, skip or compare hyp_en to ref_ja (not ideal).
        ref_for_metrics = ref_en if isinstance(ref_en, str) and len(ref_en.strip()) > 0 else ""
        bleu_s = sacrebleu.sentence_bleu(hyp_en, [ref_for_metrics]).score if ref_for_metrics else None

        # WER on words; if no ref, set None
        tx = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])
        try:
            w = wer(tx(ref_for_metrics), tx(hyp_en)) if ref_for_metrics else None
        except Exception:
            w = None

        la = local_agreement(ref_for_metrics, hyp_en) if ref_for_metrics else None
        td = token_delay_proxy(dur, len(hyp_en.split()))

        results.append({
            "audio": audio_path,
            "duration_sec": dur,
            "ref_ja": ref_ja,
            "ref_en": ref_en,
            "hyp_ja": hyp_ja,
            "hyp_en": hyp_en,
            "BLEU": bleu_s,
            "WER": w,
            "LA": la,
            "TokenDelaySecPerTok": td
        })

        progress.progress(int((len(results) / len(df)) * 100))

    elapsed = time.time() - start_time
    res_df = pd.DataFrame(results)

    st.subheader("Summary Metrics")
    # Corpus-level BLEU / mean WER / mean LA
    nonempty_mask = res_df["BLEU"].notna()
    corpus_bleu = compute_bleu(
        res_df.loc[nonempty_mask, "ref_en"].tolist(),
        res_df.loc[nonempty_mask, "hyp_en"].tolist()
    ) if nonempty_mask.any() else 0.0

    mean_wer = float(res_df["WER"].dropna().mean()) if res_df["WER"].notna().any() else 0.0
    mean_la = float(res_df["LA"].dropna().mean()) if res_df["LA"].notna().any() else 0.0
    mean_td = float(res_df["TokenDelaySecPerTok"].mean()) if len(res_df) else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Corpus BLEU", f"{corpus_bleu:.2f}")
    c2.metric("Mean WER", f"{mean_wer:.3f}")
    c3.metric("Mean LA", f"{mean_la:.3f}")
    c4.metric("Mean Token Delay (s/tok)", f"{mean_td:.3f}")
    st.caption(f"Processed {len(res_df)} items in {elapsed:.1f}s")

    st.subheader("Per-sample Results (top 200 shown)")
    st.dataframe(res_df.head(200))

    # -----------------------------
    # Plots (matplotlib, one chart per figure, no seaborn)
    # -----------------------------
    st.subheader("Charts")

    # BLEU distribution
    if res_df["BLEU"].notna().any():
        fig1 = plt.figure()
        res_df["BLEU"].dropna().plot(kind="hist", bins=20)
        plt.title("BLEU (sentence-level) distribution")
        plt.xlabel("BLEU")
        plt.ylabel("Count")
        st.pyplot(fig1)

    # WER distribution
    if res_df["WER"].notna().any():
        fig2 = plt.figure()
        res_df["WER"].dropna().plot(kind="hist", bins=20)
        plt.title("WER distribution")
        plt.xlabel("WER")
        plt.ylabel("Count")
        st.pyplot(fig2)

    # Token delay scatter vs duration
    fig3 = plt.figure()
    plt.scatter(res_df["duration_sec"], res_df["TokenDelaySecPerTok"])
    plt.title("Token Delay proxy vs Audio Duration")
    plt.xlabel("Audio duration (s)")
    plt.ylabel("Token delay (s/token)")
    st.pyplot(fig3)

    # -----------------------------
    # PDF Report
    # -----------------------------
    st.subheader("📄 Generate Detailed PDF Report")

    def save_plot_to_png(fig) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name, bbox_inches="tight", dpi=180)
        return tmp.name

    images = []
    if 'fig1' in locals():
        images.append(save_plot_to_png(fig1))
    if 'fig2' in locals():
        images.append(save_plot_to_png(fig2))
    if 'fig3' in locals():
        images.append(save_plot_to_png(fig3))

    def build_pdf(buffer: io.BytesIO):
        doc = SimpleDocTemplate(buffer, pagesize=A4, title="JP→EN Translation Report")
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("JP→EN Translation Evaluation Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Items evaluated: {len(res_df)}", styles["Normal"]))
        story.append(Paragraph(f"Corpus BLEU: {corpus_bleu:.2f}", styles["Normal"]))
        story.append(Paragraph(f"Mean WER: {mean_wer:.3f}", styles["Normal"]))
        story.append(Paragraph(f"Mean LA: {mean_la:.3f}", styles["Normal"]))
        story.append(Paragraph(f"Mean Token Delay (s/tok): {mean_td:.3f}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Sample table (first 20 rows)
        head = ["audio", "dur(s)", "ref_en", "hyp_en", "BLEU", "WER", "LA", "TD(s/tok)"]
        data = [head]
        for _, r in res_df.head(20).iterrows():
            data.append([
                os.path.basename(str(r["audio"])),
                f"{r['duration_sec']:.2f}",
                str(r["ref_en"])[:80],
                str(r["hyp_en"])[:80],
                f"{(r['BLEU'] if pd.notna(r['BLEU']) else 0):.2f}",
                f"{(r['WER'] if pd.notna(r['WER']) else 0):.3f}",
                f"{(r['LA'] if pd.notna(r['LA']) else 0):.3f}",
                f"{r['TokenDelaySecPerTok']:.3f}",
            ])

        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ALIGN', (1,1), (-1,-1), 'CENTER'),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # Plots
        for img in images:
            story.append(Paragraph(os.path.basename(img), styles["Heading4"]))
            story.append(RLImage(img, width=450, height=280))
            story.append(Spacer(1, 12))

        story.append(Paragraph(
            "Notes: BLEU computed with sacrebleu; WER with jiwer (normalized, lowercase, punctuation removed). "
            "LA is a simple token-position agreement proxy. Token Delay is a proxy (duration/|tokens|).",
            styles["Italic"]
        ))

        doc.build(story)

    buf = io.BytesIO()
    build_pdf(buf)
    st.download_button("⬇️ Download PDF Report", data=buf.getvalue(),
                       file_name="jp_en_translation_report.pdf", mime="application/pdf")

    st.success("Report generated!")
else:
    st.info("Fill the paths on the left and click **Run Evaluation**.")


# -----------------------------
# Footer note
# -----------------------------
st.caption(
    "This app uses Faster-Whisper for Japanese ASR and Helsinki-NLP opus-mt-ja-en for translation. "
    "Token Delay is a proxy for latency (not a true streaming metric)."
)
