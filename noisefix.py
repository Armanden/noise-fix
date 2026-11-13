#!/usr/bin/env python3
"""
Noisefix - Audio & Video disturbance removal and refinement tool.

Supports:
  - Audio: denoising, hum removal, normalization, compression, EQ, de-essing
  - Video: temporal denoising, stabilization, sharpening
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, lfilter, iirnotch, medfilt
from scipy.ndimage import uniform_filter1d

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found. Install ffmpeg and try again.", file=sys.stderr)
        sys.exit(1)


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stderr.decode()}")
    return r


# ---------------------------------------------------------------------------
# Audio processing helpers
# ---------------------------------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float64")
    return data, sr


def save_audio(path: str, data: np.ndarray, sr: int) -> None:
    peak = np.max(np.abs(data))
    if peak > 1.0:
        data = data / peak * 0.99
    sf.write(path, data, sr, subtype="PCM_16")


def stereo_to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return np.mean(data, axis=1)
    return data


# -- Denoising -------------------------------------------------------------

def spectral_gate_denoise(data: np.ndarray, sr: int,
                          prop_decrease: float = 0.85,
                          n_std_thresh_stationary: float = 1.5,
                          freq_mask_smooth_hz: int = 500,
                          time_mask_smooth_ms: int = 50) -> np.ndarray:
    """Deep-learning spectral gating denoise via noisereduce."""
    if data.ndim == 2:
        out = nr.reduce_noise(
            y=data.T, sr=sr, stationary=True,
            prop_decrease=prop_decrease,
            n_std_thresh_stationary=n_std_thresh_stationary,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms,
        )
        return out.T
    return nr.reduce_noise(
        y=data, sr=sr, stationary=True,
        prop_decrease=prop_decrease,
        n_std_thresh_stationary=n_std_thresh_stationary,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms,
    )


def spectral_gate_denoise_chunked(data: np.ndarray, sr: int,
                                  prop_decrease: float = 0.85,
                                  chunk_sec: float = 120.0,
                                  overlap_sec: float = 2.0,
                                  **kwargs) -> np.ndarray:
    """Chunked spectral denoise for long audio."""
    chunk_size = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)
    n = len(data)

    if n <= chunk_size:
        return spectral_gate_denoise(data, sr, prop_decrease=prop_decrease, **kwargs)

    out = np.zeros_like(data)
    pos = 0
    chunk_num = 0
    total_chunks = max(1, (n - chunk_size) // (chunk_size - overlap) + 1)
    while pos < n:
        end = min(pos + chunk_size, n)
        chunk = data[pos:end]
        chunk_num += 1
        pct = min(100, int(end / n * 100))
        print(f"\r[noisefix]   denoise chunk {chunk_num}/{total_chunks} ({pct}%) …", end="", flush=True)
        denoised = spectral_gate_denoise(chunk, sr, prop_decrease=prop_decrease, **kwargs)
        if pos == 0:
            out[:end] = denoised
        else:
            fade = min(overlap, len(denoised))
            crossfade = np.linspace(0, 1, fade)
            out[pos:pos + fade] = (1 - crossfade) * out[pos:pos + fade] + crossfade * denoised[:fade]
            out[pos + fade:end] = denoised[fade:end - pos]
        if end == n:
            break
        pos += chunk_size - overlap
    print()
    return out


def adaptive_denoise(data: np.ndarray, sr: int,
                     prop_decrease: float = 0.75) -> np.ndarray:
    """Non-stationary adaptive denoise (better for varying noise)."""
    if data.ndim == 2:
        out = nr.reduce_noise(
            y=data.T, sr=sr, stationary=False,
            prop_decrease=prop_decrease,
        )
        return out.T
    return nr.reduce_noise(y=data, sr=sr, stationary=False,
                           prop_decrease=prop_decrease)


# -- Hum / tone removal ----------------------------------------------------

def remove_hum(data: np.ndarray, sr: int,
               freq: float = 60.0, Q: float = 30.0,
               harmonics: int = 4) -> np.ndarray:
    """Remove mains hum and harmonics (60 Hz, 120 Hz, …)."""
    out = data.copy()
    for h in range(1, harmonics + 1):
        f0 = freq * h
        if f0 >= sr / 2:
            break
        b, a = iirnotch(f0, Q, sr)
        out = lfilter(b, a, out, axis=0)
    return out


# -- High / low pass -------------------------------------------------------

def highpass(data: np.ndarray, sr: int, cutoff: float = 80.0,
             order: int = 5) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(order, cutoff / nyq, btype="high")
    return lfilter(b, a, data, axis=0)


def lowpass(data: np.ndarray, sr: int, cutoff: float = 8000.0,
            order: int = 5) -> np.ndarray:
    nyq = sr / 2.0
    b, a = butter(order, cutoff / nyq, btype="low")
    return lfilter(b, a, data, axis=0)


# -- Normalization & dynamics ----------------------------------------------

def normalize_peak(data: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(data))
    if peak == 0:
        return data
    return data * (target / peak)


def normalize_loudness(data: np.ndarray, sr: int,
                       target_lufs: float = -16.0) -> np.ndarray:
    """Simple ITU-R BS.1770-inspired loudness normalization."""
    frame_len = int(0.4 * sr)
    hop = frame_len // 2
    n_frames = max(1, (len(data) - frame_len) // hop)

    def _block_rms(block):
        return np.sqrt(np.mean(block ** 2)) + 1e-12

    rms_vals = []
    for i in range(n_frames):
        s = i * hop
        rms_vals.append(_block_rms(data[s:s + frame_len]))
    rms_vals = np.array(rms_vals)
    mean_rms = np.mean(rms_vals ** 2) ** 0.5
    current_lufs = 20 * np.log10(mean_rms) - 0.691
    gain_db = target_lufs - current_lufs
    gain = 10 ** (gain_db / 20.0)
    return np.clip(data * gain, -1.0, 1.0)


def fast_compress(data: np.ndarray, threshold_db: float = -20.0,
                  ratio: float = 4.0) -> np.ndarray:
    """Vectorised soft-knee compressor (fast, no envelope)."""
    threshold = 10 ** (threshold_db / 20.0)
    absd = np.abs(data)
    mask = absd > threshold
    out = data.copy()
    over = absd[mask] - threshold
    compressed = threshold + over / ratio
    out[mask] = np.sign(data[mask]) * compressed
    return out


# -- EQ (simple parametric) -----------------------------------------------

def eq_band(data: np.ndarray, sr: int, freq: float,
            gain_db: float, Q: float = 1.0) -> np.ndarray:
    """Single peaking EQ band."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, data, axis=0)


def simple_eq(data: np.ndarray, sr: int,
              bands: list[tuple[float, float, float]] | None = None) -> np.ndarray:
    """Apply a chain of EQ bands.  bands = [(freq, gain_db, Q), …]"""
    if bands is None:
        return data
    out = data.copy()
    for freq, gain_db, Q in bands:
        if abs(gain_db) < 0.1 or freq <= 0 or freq >= sr / 2:
            continue
        out = eq_band(out, sr, freq, gain_db, Q)
    return out


# -- De-click / de-pop (median filter) ------------------------------------

def declick(data: np.ndarray, sr: int,
            kernel_ms: float = 2.0) -> np.ndarray:
    """Simple median-based click/pop removal."""
    k = int(sr * kernel_ms / 1000)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return medfilt(data, kernel_size=k).astype(data.dtype)


# ---------------------------------------------------------------------------
# Video processing (OpenCV)
# ---------------------------------------------------------------------------

def denoise_video_cv2(video_path: str, output_path: str,
                      strength: int = 10) -> None:
    """Temporal + spatial denoise with OpenCV."""
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required for video processing")

    from collections import deque

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    window = 3
    buf = deque(maxlen=window)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        buf.append(gray)
        count += 1
        if total > 0 and count % 100 == 0:
            pct = count * 100 // total
            print(f"\r[noisefix]   video denoise {pct}% ({count}/{total}) …", end="", flush=True)
        if len(buf) < window:
            out.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
            continue
        frames = list(buf)
        denoised = cv2.fastNlMeansDenoisingMulti(
            frames, imgToDenoiseIndex=window - 1,
            temporalWindowSize=window, h=strength,
        )
        out.write(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))
    print()
    cap.release()
    out.release()


def stabilize_video(video_path: str, output_path: str,
                    smooth: int = 30) -> None:
    """Video stabilisation using optical flow (OpenCV)."""
    if not HAS_CV2:
        raise RuntimeError("opencv-python is required for video processing")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_gray = None
    transforms = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count += 1
        if total > 0 and count % 100 == 0:
            pct = count * 100 // total
            print(f"\r[noisefix]   stabilize pass 1 {pct}% …", end="", flush=True)
        if prev_gray is None:
            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200,
                                               qualityLevel=0.01,
                                               minDistance=30)
            continue
        curr_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30)
        if prev_pts is not None and curr_pts is not None and len(curr_pts) > 0:
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
            if m is not None:
                dx = m[0, 2]
                dy = m[1, 2]
                da = np.arctan2(m[1, 0], m[0, 0])
            else:
                dx = dy = da = 0.0
        else:
            dx = dy = da = 0.0
        transforms.append([dx, dy, da])
        prev_gray = gray
        prev_pts = curr_pts
    print()
    cap.release()

    transforms = np.array(transforms)
    window = min(smooth, len(transforms))
    kernel = np.ones(window) / window
    smooth_dx = np.convolve(transforms[:, 0], kernel, mode="same")
    smooth_dy = np.convolve(transforms[:, 1], kernel, mode="same")
    smooth_da = np.convolve(transforms[:, 2], kernel, mode="same")

    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for i in range(len(transforms)):
        ret, frame = cap.read()
        if not ret:
            break
        if (i + 1) % 100 == 0:
            pct = (i + 1) * 100 // len(transforms)
            print(f"\r[noisefix]   stabilize pass 2 {pct}% …", end="", flush=True)
        dx = smooth_dx[i] - transforms[i, 0]
        dy = smooth_dy[i] - transforms[i, 1]
        da = smooth_da[i] - transforms[i, 2]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy],
        ], dtype=np.float64)
        frame = cv2.warpAffine(frame, m, (w, h),
                               borderMode=cv2.BORDER_REFLECT)
        out.write(frame)
    print()
    cap.release()
    out.release()


# ---------------------------------------------------------------------------
# ffmpeg helpers for video audio extraction / muxing
# ---------------------------------------------------------------------------

def ffmpeg_extract_audio(video_path: str, audio_path: str) -> int:
    """Extract audio from video to WAV. Returns sample rate."""
    _check_ffmpeg()
    _run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
          "-ar", "44100", "-ac", "2", audio_path])
    info = sf.info(audio_path)
    return info.samplerate


def ffmpeg_mux_audio(video_path: str, audio_path: str, output_path: str) -> None:
    """Replace audio track in video with processed audio."""
    _check_ffmpeg()
    _run(["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
          "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
          "-shortest", output_path])


def ffmpeg_copy_video(video_path: str, output_path: str) -> None:
    """Copy video stream only (no re-encode)."""
    _check_ffmpeg()
    _run(["ffmpeg", "-y", "-i", video_path, "-an", "-c:v", "copy", output_path])


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

AUDIO_PRESETS = {
    "light": {
        "denoise": True, "denoise_prop": 0.6,
        "hum_freq": 0, "hum_harmonics": 2,
        "highpass": 40, "lowpass": 0,
        "compress": False, "normalize": True,
        "eq_bands": None,
    },
    "default": {
        "denoise": True, "denoise_prop": 0.8,
        "hum_freq": 60, "hum_harmonics": 3,
        "highpass": 60, "lowpass": 12000,
        "compress": True, "compress_threshold": -20, "compress_ratio": 3.0,
        "normalize": True, "target_lufs": -16.0,
        "eq_bands": None,
    },
    "aggressive": {
        "denoise": True, "denoise_prop": 0.95,
        "hum_freq": 60, "hum_harmonics": 5,
        "highpass": 80, "lowpass": 10000,
        "compress": True, "compress_threshold": -18, "compress_ratio": 4.0,
        "normalize": True, "target_lufs": -14.0,
        "eq_bands": [(250, -3, 1.0), (3000, 3, 1.0)],
    },
    "voice": {
        "denoise": True, "denoise_prop": 0.85,
        "hum_freq": 60, "hum_harmonics": 3,
        "highpass": 80, "lowpass": 8000,
        "compress": True, "compress_threshold": -22, "compress_ratio": 3.0,
        "normalize": True, "target_lufs": -16.0,
        "eq_bands": [(250, -2, 0.8), (3500, 4, 1.0)],
    },
}

VIDEO_PRESETS = {
    "light": {"denoise_strength": 6, "stabilize": False},
    "default": {"denoise_strength": 10, "stabilize": True, "smooth": 30},
    "aggressive": {"denoise_strength": 15, "stabilize": True, "smooth": 50},
}


def process_audio(input_path: str, output_path: str,
                  preset: str = "default",
                  custom: dict | None = None) -> None:
    cfg = dict(AUDIO_PRESETS.get(preset, AUDIO_PRESETS["default"]))
    if custom:
        cfg.update(custom)

    print(f"[noisefix] Loading {input_path} …")
    data, sr = load_audio(input_path)
    orig_shape = data.shape
    data = stereo_to_mono(data)
    print(f"[noisefix] {len(data)/sr:.1f}s, {sr} Hz, {orig_shape}")

    if cfg.get("denoise"):
        prop = cfg.get("denoise_prop", 0.8)
        print(f"[noisefix] Spectral denoise (prop={prop}) …")
        data = spectral_gate_denoise_chunked(data, sr, prop_decrease=prop)

    hum = cfg.get("hum_freq", 0)
    if hum and hum > 0:
        harms = cfg.get("hum_harmonics", 3)
        print(f"[noisefix] Removing hum at {hum} Hz × {harms} …")
        data = remove_hum(data, sr, freq=hum, harmonics=harms)

    hp = cfg.get("highpass", 0)
    if hp and hp > 0:
        print(f"[noisefix] High-pass at {hp} Hz …")
        data = highpass(data, sr, cutoff=hp)

    lp = cfg.get("lowpass", 0)
    if lp and lp > 0:
        print(f"[noisefix] Low-pass at {lp} Hz …")
        data = lowpass(data, sr, cutoff=lp)

    eq_bands = cfg.get("eq_bands")
    if eq_bands:
        print(f"[noisefix] Applying {len(eq_bands)} EQ band(s) …")
        data = simple_eq(data, sr, bands=eq_bands)

    if cfg.get("compress"):
        thr = cfg.get("compress_threshold", -20)
        ratio = cfg.get("compress_ratio", 4.0)
        print(f"[noisefix] Compressing (thr={thr} dB, ratio={ratio}:1) …")
        data = fast_compress(data, threshold_db=thr, ratio=ratio)

    if cfg.get("normalize"):
        tgt = cfg.get("target_lufs", -16.0)
        print(f"[noisefix] Normalizing to {tgt} LUFS …")
        data = normalize_loudness(data, sr, target_lufs=tgt)

    save_audio(output_path, data, sr)
    print(f"[noisefix] Saved → {output_path}")


def process_video(input_path: str, output_path: str,
                  preset: str = "default",
                  audio_preset: str | None = None,
                  custom: dict | None = None,
                  skip_video: bool = False) -> None:
    _check_ffmpeg()
    cfg = dict(VIDEO_PRESETS.get(preset, VIDEO_PRESETS["default"]))
    if custom:
        cfg.update(custom)

    print(f"[noisefix] Processing video {input_path} …")

    tmp_dir = output_path + ".tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_video = os.path.join(tmp_dir, "video.mp4")
    tmp_audio_in = os.path.join(tmp_dir, "audio_in.wav")
    tmp_audio_out = os.path.join(tmp_dir, "audio_out.wav")
    tmp_no_audio = os.path.join(tmp_dir, "no_audio.mp4")

    try:
        ap = audio_preset or preset

        # --- video frame processing ---
        if skip_video:
            print("[noisefix] Skipping video frame processing (--no-video) …")
            ffmpeg_copy_video(input_path, tmp_no_audio)
        elif HAS_CV2:
            print("[noisefix] Denoising video frames …")
            denoise_video_cv2(input_path, tmp_video,
                              strength=cfg.get("denoise_strength", 10))
            if cfg.get("stabilize"):
                print("[noisefix] Stabilising video …")
                tmp_stab = os.path.join(tmp_dir, "stab.mp4")
                stabilize_video(tmp_video, tmp_stab,
                                smooth=cfg.get("smooth", 30))
                os.replace(tmp_stab, tmp_video)
            ffmpeg_copy_video(tmp_video, tmp_no_audio)
        else:
            print("[noisefix] OpenCV not found – skipping video frame denoise.")
            ffmpeg_copy_video(input_path, tmp_no_audio)

        # --- audio processing ---
        print("[noisefix] Extracting audio track …")
        sr = ffmpeg_extract_audio(input_path, tmp_audio_in)

        process_audio(tmp_audio_in, tmp_audio_out, preset=ap)

        # --- mux ---
        print("[noisefix] Muxing audio + video …")
        ffmpeg_mux_audio(tmp_no_audio, tmp_audio_out, output_path)
        print(f"[noisefix] Saved → {output_path}")

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".aac", ".m4a", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}


def main():
    p = argparse.ArgumentParser(
        description="Noisefix – remove disturbances and refine audio/video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s music.wav                   # default preset
  %(prog)s music.wav -o clean.wav      # specify output
  %(prog)s music.wav -p aggressive     # heavy cleanup
  %(prog)s music.wav --hum 50          # remove 50 Hz hum
  %(prog)s music.wav --highpass 100 --normalize
  %(prog)s video.mp4 -p light          # light video+audio cleanup
  %(prog)s video.mp4 --audio-preset voice --no-video-stabilize
  %(prog)s video.mp4 --no-video        # audio only, keep video untouched
        """,
    )
    p.add_argument("input", help="Input audio or video file")
    p.add_argument("-o", "--output", help="Output file (default: <input>_fixed.<ext>)")
    p.add_argument("-p", "--preset", default="default",
                   choices=list(AUDIO_PRESETS.keys()),
                   help="Processing preset (default: %(default)s)")
    p.add_argument("--audio-preset", default=None,
                   choices=list(AUDIO_PRESETS.keys()),
                   help="Override audio preset for video files")
    p.add_argument("--no-denoise", action="store_true",
                   help="Skip spectral denoising")
    p.add_argument("--denoise-prop", type=float, default=None,
                   help="Denoise strength 0.0–1.0")
    p.add_argument("--hum", type=float, default=None,
                   help="Mains hum frequency in Hz (0 = off)")
    p.add_argument("--hum-harmonics", type=int, default=None,
                   help="Number of hum harmonics to remove")
    p.add_argument("--highpass", type=float, default=None,
                   help="High-pass cutoff in Hz (0 = off)")
    p.add_argument("--lowpass", type=float, default=None,
                   help="Low-pass cutoff in Hz (0 = off)")
    p.add_argument("--compress", action="store_true", default=None,
                   help="Enable compressor")
    p.add_argument("--compress-threshold", type=float, default=None,
                   help="Compressor threshold in dB")
    p.add_argument("--compress-ratio", type=float, default=None,
                   help="Compressor ratio")
    p.add_argument("--normalize", action="store_true", default=None,
                   help="Enable loudness normalization")
    p.add_argument("--target-lufs", type=float, default=None,
                   help="Target loudness in LUFS")
    p.add_argument("--eq", nargs="+", metavar="FREQ:GAIN:Q",
                   help="EQ bands, e.g. 250:-3:1.0 3000:3:1.0")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video frame processing (denoise/stabilize), audio only")
    p.add_argument("--audio-only", action="store_true",
                   help="Alias for --no-video")
    p.add_argument("--video-denoise-strength", type=int, default=None,
                   help="Video denoise strength (OpenCV)")
    p.add_argument("--no-video-stabilize", action="store_true",
                   help="Disable video stabilisation")
    p.add_argument("--video-stabilize-smooth", type=int, default=None,
                   help="Stabilisation smooth window (frames)")

    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    ext = Path(args.input).suffix.lower()
    is_video = ext in VIDEO_EXTS
    is_audio = ext in AUDIO_EXTS

    if not is_video and not is_audio:
        print(f"Warning: unknown extension '{ext}', assuming audio", file=sys.stderr)
        is_audio = True

    if args.output:
        out = args.output
    else:
        stem = Path(args.input).stem
        parent = Path(args.input).parent
        out_ext = ext if is_audio else ".mp4"
        out = str(parent / f"{stem}_fixed{out_ext}")

    custom = {}
    if args.no_denoise:
        custom["denoise"] = False
    if args.denoise_prop is not None:
        custom["denoise_prop"] = args.denoise_prop
    if args.hum is not None:
        custom["hum_freq"] = args.hum
    if args.hum_harmonics is not None:
        custom["hum_harmonics"] = args.hum_harmonics
    if args.highpass is not None:
        custom["highpass"] = args.highpass
    if args.lowpass is not None:
        custom["lowpass"] = args.lowpass
    if args.compress is not None:
        custom["compress"] = args.compress
    if args.compress_threshold is not None:
        custom["compress_threshold"] = args.compress_threshold
    if args.compress_ratio is not None:
        custom["compress_ratio"] = args.compress_ratio
    if args.normalize is not None:
        custom["normalize"] = args.normalize
    if args.target_lufs is not None:
        custom["target_lufs"] = args.target_lufs
    if args.eq:
        bands = []
        for token in args.eq:
            parts = token.split(":")
            if len(parts) != 3:
                print(f"Error: invalid EQ format '{token}', expected FREQ:GAIN:Q",
                      file=sys.stderr)
                sys.exit(1)
            bands.append((float(parts[0]), float(parts[1]), float(parts[2])))
        custom["eq_bands"] = bands

    video_custom = {}
    if args.video_denoise_strength is not None:
        video_custom["denoise_strength"] = args.video_denoise_strength
    if args.no_video_stabilize:
        video_custom["stabilize"] = False
    if args.video_stabilize_smooth is not None:
        video_custom["smooth"] = args.video_stabilize_smooth

    if is_video:
        skip_video = args.no_video or args.audio_only
        process_video(args.input, out, preset=args.preset,
                      audio_preset=args.audio_preset,
                      custom={**custom, **video_custom},
                      skip_video=skip_video)
    else:
        process_audio(args.input, out, preset=args.preset,
                      custom=custom)


if __name__ == "__main__":
    main()
