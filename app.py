import io
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import streamlit as st

# Import core DSP logic from stem_processor.py
try:
    from stem_processor import PRESETS, process_audio_file
except ImportError:
    st.error(
        "Could not import 'stem_processor.py'. Ensure 'stem_processor.py' is in the same directory as 'app.py'."
    )
    st.stop()

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stem Smart Repair & Restorer",
    page_icon="🎛️",
    layout="wide",
)

st.title("🎛️ AI Stem Smart Repair & Restorer")
st.markdown(
    """
Restore, clean, and re-balance AI-generated or degraded audio stems. 
Upload individual WAV or MP3 files, fine-tune DSP parameters per stem type, and export high-quality 24-bit PCM WAV outputs.
"""
)

st.sidebar.header("⚙️ Global Settings")
export_format = st.sidebar.selectbox("Export Subtype", ["PCM_24", "PCM_16", "FLOAT"], index=0)

# -----------------------------------------------------------------------------
# File Upload Section
# -----------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Audio Stems (WAV, MP3, FLAC)",
    type=["wav", "mp3", "flac"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.subheader("1. Configure Stem Processing Parameters")
    
    stem_configs = {}
    
    # Render stem configuration UI
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        filename_lower = filename.lower()
        
        # Automatic Preset Detection based on filename keywords
        detected_preset = "General / Master"
        for key in PRESETS.keys():
            if key in filename_lower:
                detected_preset = key
                break

        with st.expander(f"🎚️ {filename}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                selected_preset = st.selectbox(
                    f"Preset for {filename}",
                    options=list(PRESETS.keys()),
                    index=list(PRESETS.keys()).index(detected_preset),
                    key=f"preset_{filename}",
                )
                
                # Fetch baseline parameters from selected preset
                preset_params = PRESETS[selected_preset].copy()

                hp_cutoff = st.slider(
                    "High-Pass Filter (Hz)",
                    0, 200, preset_params.get("hp_cutoff", 30),
                    key=f"hp_{filename}"
                )
                lp_cutoff = st.slider(
                    "Low-Pass Filter (Hz)",
                    8000, 22000, preset_params.get("lp_cutoff", 20000),
                    key=f"lp_{filename}"
                )

            with col2:
                deharm_gain = st.slider(
                    "De-harshness / Mid Dip (dB)",
                    -6.0, 0.0, preset_params.get("deharm_gain", -1.5), step=0.1,
                    key=f"dh_{filename}"
                )
                target_peak = st.slider(
                    "Target Peak Normalization (dBFS)",
                    -12.0, 0.0, preset_params.get("target_peak", -1.0), step=0.5,
                    key=f"peak_{filename}"
                )
                apply_limiting = st.checkbox(
                    "Enable Peak Limiter",
                    value=preset_params.get("apply_limiting", True),
                    key=f"lim_{filename}"
                )

            # Store per-file configuration
            stem_configs[filename] = {
                "file_obj": uploaded_file,
                "params": {
                    "hp_cutoff": hp_cutoff,
                    "lp_cutoff": lp_cutoff,
                    "deharm_gain": deharm_gain,
                    "target_peak": target_peak,
                    "apply_limiting": apply_limiting,
                    "subtype": export_format,
                },
            }

    st.markdown("---")
    st.subheader("2. Run Batch Stem Repair")
    
    if st.button("🚀 Process All Stems", type="primary"):
        processed_results = []
        
        # Use temporary directory context to handle file operations safely
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            out_dir = Path(temp_dir) / "out"
            raw_dir.mkdir()
            out_dir.mkdir()

            # Save uploaded bytes to temporary files
            file_tasks = []
            for filename, cfg in stem_configs.items():
                raw_path = raw_dir / filename
                out_path = out_dir / f"repaired_{filename}"
                
                with open(raw_path, "wb") as f:
                    f.write(cfg["file_obj"].getbuffer())
                    
                file_tasks.append((str(raw_path), str(out_path), cfg["params"], filename))

            # Helper function for concurrent processing
            def worker(task):
                input_p, output_p, params, fname = task
                process_audio_file(input_path=input_p, output_path=output_p, **params)
                
                # Read processed output bytes for preview and download
                with open(output_p, "rb") as f:
                    processed_bytes = f.read()
                return fname, processed_bytes

            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Processing stems in parallel...")

            # Run DSP tasks concurrently across available CPU threads
            completed = 0
            total = len(file_tasks)
            processed_data = {}

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(worker, task) for task in file_tasks]
                for future in futures:
                    fname, p_bytes = future.result()
                    processed_data[fname] = p_bytes
                    completed += 1
                    progress_bar.progress(completed / total)

            status_text.success("✅ Processing complete!")

            # Create in-memory ZIP package
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, p_bytes in processed_data.items():
                    zf.writestr(f"repaired_{fname}", p_bytes)
            
            zip_buffer.seek(0)

            # Download Section
            st.markdown("### 3. Download & Preview")
            st.download_button(
                label="📦 Download All Repaired Stems (.ZIP)",
                data=zip_buffer,
                file_name="repaired_stems_package.zip",
                mime="application/zip",
                type="primary"
            )

            # Audio Player Previews
            st.markdown("#### Audio Comparisons")
            for filename, p_bytes in processed_data.items():
                st.markdown(f"**{filename}**")
                col_orig, col_proc = st.columns(2)
                
                with col_orig:
                    st.caption("Original")
                    st.audio(stem_configs[filename]["file_obj"], format="audio/wav")
                    
                with col_proc:
                    st.caption("Repaired")
                    st.audio(p_bytes, format="audio/wav")

else:
    st.info("👆 Upload your audio stems above to get started.")
