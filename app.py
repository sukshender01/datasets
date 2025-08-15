import os
import pandas as pd
import zipfile
import streamlit as st
from collections import defaultdict

st.set_page_config(page_title="Dataset Audio Link Checker", layout="wide")

st.title("🎧 Audio Dataset Link Checker")
st.write("Upload your dataset folder (as a `.zip`) containing `.tsv` metadata and `.mp3` files to check linking consistency.")

# File uploader for dataset ZIP
uploaded_zip = st.file_uploader("Upload Dataset ZIP", type=["zip"])

if uploaded_zip:
    # Extract ZIP
    extract_path = "dataset_temp"
    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    st.success("✅ Dataset extracted successfully.")

    # Find all TSV files
    tsv_files = []
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".tsv"):
                tsv_files.append(os.path.join(root, file))

    if not tsv_files:
        st.error("❌ No TSV files found in the dataset.")
    else:
        # Find all MP3 files
        all_audio_files = set()
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith(".mp3"):
                    rel_path = os.path.relpath(os.path.join(root, file), extract_path)
                    all_audio_files.add(rel_path.replace("\\", "/"))

        tsv_audio_refs = defaultdict(set)
        missing_audio = defaultdict(list)

        # Check each TSV file
        for tsv_file in tsv_files:
            df = pd.read_csv(tsv_file, sep="\t", dtype=str)

            if "path" not in df.columns:
                st.warning(f"⚠️ `{os.path.basename(tsv_file)}` missing 'path' column. Skipping.")
                continue

            for _, row in df.iterrows():
                audio_path = row["path"].strip()
                tsv_audio_refs[os.path.basename(tsv_file)].add(audio_path)
                if audio_path not in all_audio_files:
                    missing_audio[os.path.basename(tsv_file)].append(audio_path)

        # Prepare results
        st.subheader("📄 Evaluation Summary")
        summary_data = []
        for tsv_name in tsv_audio_refs:
            summary_data.append({
                "TSV File": tsv_name,
                "Total Audio References": len(tsv_audio_refs[tsv_name]),
                "Missing Audio Files": len(missing_audio[tsv_name])
            })
        st.dataframe(pd.DataFrame(summary_data))

        # Missing file details
        st.subheader("🚫 Missing Audio Files")
        for tsv_name, missing_list in missing_audio.items():
            if missing_list:
                st.write(f"**{tsv_name}** - {len(missing_list)} missing")
                st.write(missing_list[:10])  # show first 10 only

        # Unused files
        st.subheader("📌 Unused Audio Files")
        used_audio = set().union(*tsv_audio_refs.values())
        unused_audio = all_audio_files - used_audio
        st.write(f"Total Unused Files: {len(unused_audio)}")
        if unused_audio:
            st.write(list(unused_audio)[:10])  # show first 10 only

        # Option to download results
        st.subheader("⬇ Download Full Report")
        report_df = pd.DataFrame({
            "TSV File": list(tsv_audio_refs.keys()) * 2,
            "Type": ["Missing"] * len(tsv_audio_refs) + ["Unused"] * len(tsv_audio_refs),
            "Files": [missing_audio[tsv] for tsv in tsv_audio_refs] +
                     [list(unused_audio) for _ in tsv_audio_refs]
        })
        csv_path = "dataset_link_report.csv"
        report_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button("Download CSV Report", f, file_name="dataset_link_report.csv")

