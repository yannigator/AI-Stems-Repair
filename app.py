import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import soundfile as sf
import streamlit as st

# Import your DSP pipeline functions from your main processing module
# (Assuming your pipeline functions are in stem_processor.py or defined locally)
from stem_processor import repair_ai_vocal, combine_stems, STEM_PRESETS, DEFAULT_PRESET

st.set_page_config(
    page_title="Stem Smart Repair & Restorer",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ AI Stem Smart Repair & Restorer")
st.markdown("Upload your raw audio stems (`.wav`), fine-tune DSP parameters per track, and export a clean processed bundle.")

# Sidebar - Global Settings
st.sidebar.header("⚙️ Global Settings")
export_combined = st.sidebar.checkbox("Generate FULL_MIX_COMBINED.wav", value=True)
export_zip = st.sidebar.checkbox("Package into ZIP download", value=True)

# File Uploader
uploaded_files = st.file_uploader(
    "Upload Stem WAV Files", 
    type=["wav"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("🎚️ Stem Configuration")
    
    # Storage for per-stem parameters
    stem_configs = {}
    
    # Display configuration columns or expanders for each uploaded stem
    for file in uploaded_files:
        filename = file.name
        filename_lower = filename.lower()
        
        # Match preset
        matched_key = None
        for key in sorted(STEM_PRESETS.keys(), key=len, reverse=True):
            if key in filename_lower:
                matched_key = key
                break
        
        default_params = STEM_PRESETS.get(matched_key, DEFAULT_PRESET)
        preset_name = matched_key.upper() if matched_key else "DEFAULT"
        
        with st.expander(f"🔊 {filename} — [{preset_name}]", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**De-Esser & High-Pass**")
                deesser_thresh = st.slider("De-Esser Threshold (dB)", -40.0, 0.0, float(default_params["deesser_threshold"]), key=f"de_{filename}")
                deesser_ratio = st.slider("De-Esser Ratio", 1.0, 10.0, float(default_params["deesser_ratio"]), key=f"dr_{filename}")
                deesser_mix = st.slider("De-Esser Mix", 0.0, 1.0, float(default_params["deesser_mix"]), key=f"dm_{filename}")
            
            with col2:
                st.markdown("**Body & Warmth**")
                warmth_drive = st.slider("Warmth Drive (dB)", 0.0, 6.0, float(default_params["warmth_drive"]), key=f"wd_{filename}")
                warmth_mix = st.slider("Warmth Mix", 0.0, 1.0, float(default_params["warmth_mix"]), key=f"wm_{filename}")
            
            with col3:
                st.markdown("**Stereo & Dynamics**")
                widener_side = st.slider("Side Gain (dB)", 0.0, 6.0, float(default_params["widener_side_gain"]), key=f"ws_{filename}")
                widener_mix = st.slider("Widener Mix", 0.0, 1.0, float(default_params["widener_mix"]), key=f"wx_{filename}")
                volume_ride = st.checkbox("Enable Gentle Volume Rider", value=bool(default_params["volume_ride"]), key=f"vr_{filename}")
            
            stem_configs[filename] = {
                "file_obj": file,
                "params": {
                    "deesser_threshold": deesser_thresh,
                    "deesser_ratio": deesser_ratio,
                    "deesser_mix": deesser_mix,
                    "warmth_drive": warmth_drive,
                    "warmth_mix": warmth_mix,
                    "widener_side_gain": widener_side,
                    "widener_mix": widener_mix,
                    "volume_ride": volume_ride,
                }
            }

    st.markdown("---")
    if st.button("🚀 Process All Stems", type="primary"):
        with st.spinner("Processing audio stems..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                raw_dir = temp_path / "raw"
                out_dir = temp_path / "processed"
                raw_dir.mkdir()
                out_dir.mkdir()
                
                processed_paths = []
                
                # 1. Process individual stems
                for filename, config in stem_configs.items():
                    raw_file_path = raw_dir / filename
                    out_file_path = out_dir / f"processed_{filename}"
                    
                    # Save uploaded file bytes to temp folder
                    with open(raw_file_path, "wb") as f:
                        f.write(config["file_obj"].getbuffer())
                    
                    # Run DSP pipeline
                    repair_ai_vocal(
                        input_path=str(raw_file_path),
                        output_path=str(out_file_path),
                        export_format="wav",
                        **config["params"]
                    )
                    processed_paths.append(str(out_file_path))
                
                # 2. Combine stems if requested
                combined_path = None
                if export_combined and len(processed_paths) > 0:
                    combined_path = str(out_dir / "FULL_MIX_COMBINED.wav")
                    combine_stems(processed_paths, combined_path)
                
                # 3. Create ZIP package
                zip_buffer_path = temp_path / "processed_stems.zip"
                with zipfile.ZipFile(zip_buffer_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for p in processed_paths:
                        zipf.write(p, arcname=Path(p).name)
                    if combined_path and os.path.exists(combined_path):
                        zipf.write(combined_path, arcname="FULL_MIX_COMBINED.wav")
                
                st.success("✅ Audio processing complete!")
                
                # 4. Audio Players & Downloads
                st.subheader("🎧 Audio Preview & Downloads")
                
                if combined_path and os.path.exists(combined_path):
                    st.markdown("**Full Mix Combined Preview**")
                    st.audio(combined_path, format="audio/wav")
                
                with open(zip_buffer_path, "rb") as zf:
                    st.download_button(
                        label="📦 Download All Processed Stems (ZIP)",
                        data=zf.read(),
                        file_name="Processed_Stems_Package.zip",
                        mime="application/zip",
                        type="primary"
                    )
