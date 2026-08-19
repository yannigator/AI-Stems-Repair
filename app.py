import os
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import butter, sosfilt


# =====================================================================
# 1. AUDIO PROCESSING FUNCTIONS (BUG-PROOFED DSP PIPELINE)
# =====================================================================
def apply_dynamic_de_esser(
    audio: np.ndarray,
    sr: int,
    freq_range: tuple = (5000, 8500),
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 1.0,
    release_ms: float = 50.0,
    mix: float = 1.0,
    solo: bool = False,
) -> np.ndarray:
    if np.all(audio == 0):
        return audio

    sos_bp = butter(2, freq_range, btype="bandpass", fs=sr, output="sos")
    attack_coeff = np.exp(-1.0 / (sr * (max(attack_ms, 0.1) / 1000.0)))
    release_coeff = np.exp(-1.0 / (sr * (max(release_ms, 0.1) / 1000.0)))

    def process_channel(ch: np.ndarray) -> np.ndarray:
        sibilance_band = sosfilt(sos_bp, ch)
        abs_band = np.abs(sibilance_band)

        # Isolated envelope follower per channel
        envelope = np.zeros_like(abs_band)
        curr_env = 0.0
        for i in range(len(abs_band)):
            val = abs_band[i]
            if val > curr_env:
                curr_env = attack_coeff * curr_env + (1.0 - attack_coeff) * val
            else:
                curr_env = release_coeff * curr_env + (1.0 - release_coeff) * val
            envelope[i] = curr_env

        env_db = 20 * np.log10(envelope + 1e-12)
        gain_db = np.zeros_like(env_db)
        over_thresh = env_db > threshold_db

        if ratio > 1.0:
            gain_db[over_thresh] = -(env_db[over_thresh] - threshold_db) * (
                1.0 - (1.0 / ratio)
            )
        gain_db = np.clip(gain_db, -30.0, 0.0)
        gain_linear = 10 ** (gain_db / 20.0)

        processed = ch - (sibilance_band * (1.0 - gain_linear))

        if solo:
            return sibilance_band * (1.0 - gain_linear)

        return ch * (1.0 - mix) + processed * mix

    if audio.ndim == 1:
        return process_channel(audio)
    return np.vstack([process_channel(audio[i]) for i in range(audio.shape[0])])


def apply_vocal_body_enhancer(
    audio: np.ndarray,
    sr: int,
    drive_db: float = 2.0,
    mix: float = 0.5,
    freq_low: float = 150.0,
    freq_high: float = 600.0,
    adaptive: bool = True,
) -> np.ndarray:
    if mix <= 0.0 or np.all(audio == 0):
        return audio

    drive = 10 ** (drive_db / 20.0)
    sos_warmth = butter(
        2, [freq_low, freq_high], btype="bandpass", fs=sr, output="sos"
    )

    def process_channel(ch: np.ndarray) -> np.ndarray:
        warmth_band = sosfilt(sos_warmth, ch)
        saturated = np.tanh(warmth_band * drive) / max(drive, 1e-6)

        if adaptive:
            envelope = np.abs(ch)
            sigma_val = max(int(sr * 0.01), 1)
            envelope = gaussian_filter1d(envelope, sigma=sigma_val)
            max_env = np.max(envelope)
            if max_env > 0:
                envelope = envelope / max_env
            saturated = saturated * (0.3 + 0.7 * envelope)

        processed = ch + saturated * 0.25
        return ch * (1.0 - mix) + processed * mix

    if audio.ndim == 1:
        return process_channel(audio)
    return np.vstack([process_channel(audio[i]) for i in range(audio.shape[0])])


def apply_ms_stereo_widener(
    audio: np.ndarray,
    side_gain_db: float = 1.5,
    center_focus: float = 0.0,
    mix: float = 1.0,
) -> np.ndarray:
    if audio.ndim == 1 or audio.shape[0] < 2 or mix <= 0.0:
        return audio

    left, right = audio[0], audio[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    side_scale = 10 ** (side_gain_db / 20.0)
    side = side * side_scale

    if center_focus != 0.0:
        mid = mid * (1.0 + center_focus * 0.1)

    new_left = mid + side
    new_right = mid - side
    processed = np.vstack([new_left, new_right])

    if mix < 1.0:
        dry = np.vstack([left, right])
        return dry * (1.0 - mix) + processed * mix

    return processed


def apply_gentle_volume_ride(
    audio: np.ndarray,
    sr: int,
    window_ms: float = 500.0,
    max_gain_db: float = 3.0,
) -> np.ndarray:
    if np.all(audio == 0):
        return audio

    window_samples = max(int(sr * (window_ms / 1000.0)), 1)

    def process_channel(ch: np.ndarray) -> np.ndarray:
        squared = ch**2
        mean_squared = uniform_filter1d(squared, size=window_samples)
        rms = np.sqrt(np.maximum(mean_squared, 0.0) + 1e-12)

        active_rms = rms[rms > 0.001]
        if len(active_rms) > 0:
            median_rms = np.median(active_rms)
        else:
            median_rms = np.median(rms)

        if median_rms <= 1e-12:
            return ch

        target_db = 20 * np.log10(median_rms + 1e-12) - 3.0
        rms_db = 20 * np.log10(rms + 1e-12)
        gain_db = target_db - rms_db
        gain_db = np.clip(gain_db, -max_gain_db, max_gain_db)

        sigma_val = max(int(sr * 0.05), 1)
        gain_db = gaussian_filter1d(gain_db, sigma=sigma_val)
        gain_linear = 10 ** (gain_db / 20.0)
        return ch * gain_linear

    if audio.ndim == 1:
        return process_channel(audio)
    return np.vstack([process_channel(audio[i]) for i in range(audio.shape[0])])


def repair_ai_vocal(
    input_path: str,
    output_path: str,
    deesser_threshold: float = -20.0,
    deesser_ratio: float = 4.0,
    deesser_mix: float = 0.8,
    warmth_drive: float = 1.5,
    warmth_mix: float = 0.4,
    widener_side_gain: float = 1.2,
    widener_mix: float = 0.7,
    volume_ride: bool = True,
    export_format: str = "wav",
) -> dict:
    audio, sr = librosa.load(input_path, sr=None, mono=False)

    # Convert 1D mono array to 2D shape (1, samples) for consistent matrix operations
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)

    # 1. High-pass filter (@ 75 Hz)
    sos_hp = butter(2, 75, btype="highpass", fs=sr, output="sos")
    audio = np.vstack([sosfilt(sos_hp, audio[i]) for i in range(audio.shape[0])])

    # 2. De-esser
    audio = apply_dynamic_de_esser(
        audio,
        sr,
        threshold_db=deesser_threshold,
        ratio=deesser_ratio,
        mix=deesser_mix,
    )

    # 3. Vocal body enhancer
    audio = apply_vocal_body_enhancer(
        audio, sr, drive_db=warmth_drive, mix=warmth_mix, adaptive=True
    )

    # 4. Stereo widener
    if audio.shape[0] >= 2:
        audio = apply_ms_stereo_widener(
            audio, side_gain_db=widener_side_gain, mix=widener_mix
        )

    # 5. Volume rider
    if volume_ride:
        audio = apply_gentle_volume_ride(audio, sr, max_gain_db=3.0)

    # 6. Global peak normalization (-1.0 dBFS safety)
    max_peak = np.max(np.abs(audio))
    if max_peak > 0:
        target_linear = 10 ** (-1.0 / 20.0)
        audio = audio * (target_linear / max_peak)

    # Dynamic Range Calculation
    active_samples = np.abs(audio)[np.abs(audio) > 0.001]
    if len(active_samples) > 0:
        p95 = np.percentile(20 * np.log10(active_samples), 95)
        p5 = np.percentile(20 * np.log10(active_samples), 5)
        dynamic_range_db = float(p95 - p5)
    else:
        dynamic_range_db = 0.0

    metadata = {
        "sample_rate": sr,
        "channels": audio.shape[0],
        "duration_seconds": audio.shape[-1] / sr,
        "processed_peak_db": float(20 * np.log10(max(np.max(np.abs(audio)), 1e-12))),
        "dynamic_range_db": dynamic_range_db,
    }

    # Transpose back to (samples, channels) for soundfile export
    sf.write(
        output_path,
        audio.T,
        sr,
        subtype="PCM_24" if export_format == "wav" else None,
    )
    return metadata


# =====================================================================
# 2. STEM PRESETS DEFINITION
# =====================================================================
STEM_PRESETS = {
    "backing vocal": {
        "deesser_threshold": -20.0,
        "deesser_ratio": 4.0,
        "deesser_mix": 0.8,
        "warmth_drive": 1.2,
        "warmth_mix": 0.35,
        "widener_side_gain": 2.2,
        "widener_mix": 0.85,
        "volume_ride": True,
    },
    "vocal": {
        "deesser_threshold": -22.0,
        "deesser_ratio": 4.5,
        "deesser_mix": 0.85,
        "warmth_drive": 1.8,
        "warmth_mix": 0.45,
        "widener_side_gain": 0.5,
        "widener_mix": 0.2,
        "volume_ride": True,
    },
    "bass": {
        "deesser_threshold": 0.0,
        "deesser_ratio": 1.0,
        "deesser_mix": 0.0,
        "warmth_drive": 2.5,
        "warmth_mix": 0.5,
        "widener_side_gain": 0.0,
        "widener_mix": 0.0,
        "volume_ride": True,
    },
    "drums": {
        "deesser_threshold": -18.0,
        "deesser_ratio": 2.5,
        "deesser_mix": 0.5,
        "warmth_drive": 1.5,
        "warmth_mix": 0.3,
        "widener_side_gain": 1.0,
        "widener_mix": 0.4,
        "volume_ride": False,
    },
    "percussions": {
        "deesser_threshold": -18.0,
        "deesser_ratio": 3.0,
        "deesser_mix": 0.6,
        "warmth_drive": 1.2,
        "warmth_mix": 0.3,
        "widener_side_gain": 1.8,
        "widener_mix": 0.7,
        "volume_ride": False,
    },
    "synth": {
        "deesser_threshold": -24.0,
        "deesser_ratio": 3.0,
        "deesser_mix": 0.5,
        "warmth_drive": 1.5,
        "warmth_mix": 0.4,
        "widener_side_gain": 2.5,
        "widener_mix": 0.8,
        "volume_ride": True,
    },
    "strings": {
        "deesser_threshold": -25.0,
        "deesser_ratio": 3.0,
        "deesser_mix": 0.6,
        "warmth_drive": 1.4,
        "warmth_mix": 0.4,
        "widener_side_gain": 2.0,
        "widener_mix": 0.75,
        "volume_ride": True,
    },
}

DEFAULT_PRESET = {
    "deesser_threshold": -20.0,
    "deesser_ratio": 3.0,
    "deesser_mix": 0.5,
    "warmth_drive": 1.2,
    "warmth_mix": 0.3,
    "widener_side_gain": 1.0,
    "widener_mix": 0.5,
    "volume_ride": True,
}


# =====================================================================
# 3. HELPER: COMBINE STEMS INTO SINGLE MIX (SAFE SUMMING)
# =====================================================================
def combine_stems(processed_files: list, output_combined_path: str):
    if not processed_files:
        return

    audio_buffers = []
    sr = None

    for file_path in processed_files:
        data, current_sr = sf.read(file_path, dtype="float32")
        if sr is None:
            sr = current_sr
        elif sr != current_sr:
            # Resample on the fly if sample rates mismatch
            data = librosa.resample(data.T, orig_sr=current_sr, target_sr=sr).T

        # Force buffer shape to 2D (samples, 2)
        if data.ndim == 1:
            data = np.column_stack([data, data])
        elif data.shape[1] == 1:
            data = np.column_stack([data[:, 0], data[:, 0]])

        audio_buffers.append(data)

    max_len = max(buf.shape[0] for buf in audio_buffers)
    combined = np.zeros((max_len, 2), dtype=np.float32)

    for buf in audio_buffers:
        if buf.shape[0] < max_len:
            pad_width = ((0, max_len - buf.shape[0]), (0, 0))
            buf = np.pad(buf, pad_width, mode="constant")
        combined += buf

    max_peak = np.max(np.abs(combined))
    if max_peak > 0:
        target_linear = 10 ** (-1.0 / 20.0)
        combined = combined * (target_linear / max_peak)

    sf.write(output_combined_path, combined, sr, subtype="PCM_24")
    print(f"🎛️ Combined track exported safely to: {output_combined_path}")


# =====================================================================
# 4. MAIN BATCH PROCESSOR
# =====================================================================
def process_stem_folder(
    input_dir: str,
    output_dir: str,
    create_combined_track: bool = True,
    create_zip_file: bool = True,
    zip_filename: str = "processed_stems_package.zip",
) -> dict:

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(input_path.glob("*.wav")))
    if not wav_files:
        print(f"❌ No .wav files found in '{input_dir}'")
        return {}

    print(
        f"\n🚀 Found {len(wav_files)} WAV files to process in '{input_dir}'\n"
        + "=" * 60
    )

    processed_file_paths = []

    for file_path in wav_files:
        filename_lower = file_path.stem.lower()

        # Prioritize matching longer keys first (e.g., "backing vocal" before "vocal")
        matched_key = None
        for key in sorted(STEM_PRESETS.keys(), key=len, reverse=True):
            if key in filename_lower:
                matched_key = key
                break

        preset = STEM_PRESETS.get(matched_key, DEFAULT_PRESET)
        preset_name = matched_key.upper() if matched_key else "DEFAULT"

        out_file = output_path / f"processed_{file_path.name}"

        print(f"🔊 Processing: {file_path.name}")
        print(f"   └── Preset: [{preset_name}]")

        metadata = repair_ai_vocal(
            input_path=str(file_path),
            output_path=str(out_file),
            export_format="wav",
            **preset,
        )

        processed_file_paths.append(str(out_file))

        print(
            f"   └── Done! Peak: {metadata['processed_peak_db']:.2f} dB | "
            f"Dynamic Range: {metadata['dynamic_range_db']:.2f} dB\n"
        )

    # 1. OPTIONAL: Combine all stems into a single track
    combined_track_path = None
    if create_combined_track and len(processed_file_paths) > 0:
        combined_track_path = str(output_path / "FULL_MIX_COMBINED.wav")
        combine_stems(processed_file_paths, combined_track_path)

    # 2. OPTIONAL: Package everything into a ZIP file
    zip_path = None
    if create_zip_file:
        zip_path = str(output_path / zip_filename)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_str in processed_file_paths:
                p = Path(file_str)
                zipf.write(p, arcname=p.name)
            if combined_track_path and os.path.exists(combined_track_path):
                cp = Path(combined_track_path)
                zipf.write(cp, arcname=cp.name)

        print(f"📦 ZIP package created at: {zip_path}")

    print("=" * 60 + f"\n✅ Processing complete! Output folder: '{output_dir}'")

    return {
        "processed_stems": processed_file_paths,
        "combined_track": combined_track_path,
        "zip_package": zip_path,
    }


# =====================================================================
# 5. SCRIPT EXECUTION
# =====================================================================
if __name__ == "__main__":
    INPUT_FOLDER = "./raw_stems"
    OUTPUT_FOLDER = "./processed_stems"

    # Auto-create input folder if missing to prevent immediately failing
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    results = process_stem_folder(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        create_combined_track=True,
        create_zip_file=True,
        zip_filename="T-Meel_Stems_Processed.zip",
    )
