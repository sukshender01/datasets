import streamlit as st
import pandas as pd
from deep_translator import GoogleTranslator
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

# ---- CONFIG ----
st.set_page_config(page_title="JP→EN Translation Evaluation", layout="wide")

# ---- LOAD DATA ----
st.title("Japanese → English Translation & Evaluation Report")

csv_file = "matched_dataset.csv"
if not os.path.exists(csv_file):
    st.error(f"{csv_file} not found!")
    st.stop()

df = pd.read_csv(csv_file)

if 'sentence' not in df.columns:
    st.error("No 'sentence' column found in dataset.")
    st.stop()

st.write(f"Loaded {len(df)} rows from dataset.")

# ---- TRANSLATION ----
st.subheader("Translating Japanese → English...")
translator = GoogleTranslator(source="ja", target="en")

df['predicted_translation'] = df['sentence'].apply(lambda x: translator.translate(x))

# ---- METRICS ----
st.subheader("Calculating Metrics...")

bleu_scores, wers, la_scores, atd_scores = [], [], [], []

for _, row in df.iterrows():
    ref = row.get("reference", "") or ""  # assumed column for reference English
    hyp = row['predicted_translation']

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smoothie) if ref else None

    # WER
    w = wer(ref, hyp) if ref else None

    # Dummy LA & ATD (since no live token-by-token times)
    la = sum(1 for a, b in zip(ref.split(), hyp.split()) if a == b) / max(len(ref.split()), 1) if ref else None
    atd = abs(len(hyp.split()) - len(ref.split())) if ref else None

    bleu_scores.append(bleu)
    wers.append(w)
    la_scores.append(la)
    atd_scores.append(atd)

df['BLEU'] = bleu_scores
df['WER'] = wers
df['LA'] = la_scores
df['ATD'] = atd_scores

# ---- PLOTS ----
st.subheader("Metrics Visualization")
fig, ax = plt.subplots()
ax.plot(df['BLEU'], label="BLEU")
ax.plot(df['WER'], label="WER")
ax.plot(df['LA'], label="Local Agreement")
ax.plot(df['ATD'], label="Token Delay")
ax.set_xlabel("Sentence Index")
ax.set_ylabel("Score")
ax.legend()
st.pyplot(fig)

# ---- REPORT GENERATION ----
st.subheader("Generate Detailed PDF Report")

def generate_pdf(dataframe, filename="translation_report.pdf"):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename)
    content = []
    content.append(Paragraph("Japanese → English Translation Report", styles["Title"]))
    content.append(Spacer(1, 12))
    for _, row in dataframe.iterrows():
        content.append(Paragraph(f"JP: {row['sentence']}", styles["Normal"]))
        content.append(Paragraph(f"EN Predicted: {row['predicted_translation']}", styles["Normal"]))
        if 'reference' in dataframe.columns:
            content.append(Paragraph(f"EN Reference: {row['reference']}", styles["Normal"]))
        content.append(Paragraph(f"BLEU: {row['BLEU']}, WER: {row['WER']}, LA: {row['LA']}, ATD: {row['ATD']}", styles["Normal"]))
        content.append(Spacer(1, 12))
    doc.build(content)
    return filename

if st.button("Generate PDF Report"):
    pdf_path = generate_pdf(df)
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF", f, file_name="translation_report.pdf")

