import os
import subprocess
import tempfile

import librosa
import numpy as np

import audio_cache

SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 64
N_MFCC = 20
CLIP_SECONDS = 30
MIN_SECONDS = 8

TEMPO = getattr(librosa.feature, "tempo", None) or librosa.feature.rhythm.tempo


def decode(raw, suffix):
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", path, "-t", str(CLIP_SECONDS),
                "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-",
            ],
            capture_output=True,
            timeout=90,
        )
    finally:
        os.remove(path)
    if proc.returncode != 0 or not proc.stdout:
        return None
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if len(y) < MIN_SECONDS * SR:
        return None
    return y


def vector_from_signal(y):
    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        return None
    y = y / peak

    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    power = stft ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=SR, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=1.0)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)

    chroma = librosa.feature.chroma_stft(S=power, sr=SR).mean(axis=1)
    chroma = np.roll(chroma, -int(np.argmax(chroma)))
    chroma = chroma / (chroma.max() + 1e-9)

    contrast = librosa.feature.spectral_contrast(S=stft + 1e-2, sr=SR, fmin=100.0).mean(axis=1)

    centroid = librosa.feature.spectral_centroid(S=stft, sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=SR)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    spectral = []
    for series in (np.log1p(centroid), np.log1p(bandwidth), np.log1p(rolloff),
                   np.log10(flatness + 1e-8), zcr):
        spectral += [float(series.mean()), float(series.std())]

    rms = librosa.feature.rms(S=stft, frame_length=N_FFT)[0]
    rms_db = 20.0 * np.log10(rms + 1e-6)
    energy = [float(rms_db.mean()), float(rms_db.std())]

    onset = librosa.onset.onset_strength(S=log_mel, sr=SR, hop_length=HOP)
    tempo = float(np.atleast_1d(
        TEMPO(onset_envelope=onset, sr=SR, hop_length=HOP)
    )[0])
    centered = onset - onset.mean()
    ac = librosa.autocorrelate(centered)
    lo = int(round(60.0 * SR / (HOP * 200.0)))
    hi = min(int(round(60.0 * SR / (HOP * 60.0))), len(ac) - 1)
    clarity = float(ac[lo:hi + 1].max() / (ac[0] + 1e-9)) if hi > lo and ac[0] > 0 else 0.0
    rhythm = [
        float(np.log2(np.clip(tempo, 40.0, 240.0))),
        float(onset.mean()),
        float(onset.std()),
        clarity,
    ]

    vec = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        chroma,
        contrast,
        np.array(spectral),
        np.array(energy),
        np.array(rhythm),
    ]).astype(np.float32)
    if vec.shape[0] != audio_cache.DIM or not np.all(np.isfinite(vec)):
        return None
    return vec


def extract_vector(raw, suffix):
    y = decode(raw, suffix)
    if y is None:
        return None
    return vector_from_signal(y)
