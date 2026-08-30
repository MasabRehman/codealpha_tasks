"""
Task 2: Speech Signal Processing & Feature Extraction Pipeline
Extracts MFCCs, Mel-Spectrograms, Chroma, Spectral Contrast, ZCR, and RMS Energy.
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

def load_audio(file_path: str, target_sr: int = 22050):
    """
    Loads audio file as mono float32 array in [-1.0, 1.0].
    """
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception:
        # Fallback using scipy.io.wavfile
        sr, data = wavfile.read(file_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        return data, sr

def extract_features_from_audio(audio: np.ndarray, sr: int = 22050, n_mfcc: int = 40, max_len: int = 150):
    """
    Extracts multi-domain speech features:
    1. MFCCs (Mel-Frequency Cepstral Coefficients)
    2. Mel-Spectrogram
    3. Chroma Features
    4. Spectral Contrast
    5. Zero Crossing Rate (ZCR)
    6. Root Mean Square (RMS) Energy
    
    Returns:
    - fixed_length_mfcc: 2D array [n_mfcc, max_len] suitable for 1D/2D CNNs & LSTMs
    - feature_vector: 1D aggregated statistics vector (mean + std)
    """
    try:
        import librosa
        # 1. MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # 3. Mel-Spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        
        # 6. RMS
        rms = librosa.feature.rms(y=audio)
        
        # Aggregate 1D statistical summary
        feature_vector = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(contrast, axis=1), np.std(contrast, axis=1),
            np.mean(zcr, axis=1), np.std(zcr, axis=1),
            np.mean(rms, axis=1), np.std(rms, axis=1)
        ])
        
    except ImportError:
        # Pure scipy / numpy fallback for feature computation
        # FFT computation
        freqs, times, Sxx = signal.spectrogram(audio, fs=sr, nperseg=512, noverlap=256)
        log_spec = np.log(Sxx + 1e-6)
        
        # Synthetic DCT-based cepstrum
        dct_spec = np.zeros((n_mfcc, log_spec.shape[1]))
        for k in range(n_mfcc):
            dct_spec[k, :] = np.sum(log_spec * np.cos(np.pi * k * (np.arange(log_spec.shape[0])[:, None] + 0.5) / log_spec.shape[0]), axis=0)
            
        mfcc = dct_spec
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0
        rms = np.sqrt(np.mean(audio**2))
        
        feature_vector = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            [zcr, rms]
        ])
        
    # Pad or trim 2D MFCC sequence to max_len
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        padded_mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        padded_mfcc = mfcc[:, :max_len]
        
    return padded_mfcc.astype(np.float32), feature_vector.astype(np.float32)

def extract_features_from_file(file_path: str, sr: int = 22050, n_mfcc: int = 40, max_len: int = 150):
    """
    Helper to load a WAV file and extract features directly.
    """
    audio, sample_rate = load_audio(file_path, target_sr=sr)
    return extract_features_from_audio(audio, sr=sample_rate, n_mfcc=n_mfcc, max_len=max_len)
