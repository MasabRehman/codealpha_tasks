"""
Task 2: Speech Audio Dataset Generator & Acoustic Synthesizer
Generates synthetic speech audio clips with characteristic pitch, formant, and energy variations
for 6 emotion classes: Angry, Happy, Sad, Neutral, Fear, Surprise.
Compatible with RAVDESS/TESS/EMO-DB dataset formats.
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise"]

def synthesize_emotional_wave(emotion: str, duration: float = 2.5, sr: int = 22050, seed: int = None) -> np.ndarray:
    """
    Synthesizes an audio waveform mimicking the acoustic and prosodic properties of a given emotion:
    - Angry: High fundamental frequency (F0), heavy harmonics, high energy, sharp attack.
    - Happy: Elevated pitch with rising melody/vibrato, energetic cadence.
    - Sad: Low pitch, slow downward glide, low energy, long decay.
    - Neutral: Moderate constant pitch, steady rhythmic envelope.
    - Fear: Fast erratic jitter, high pitch tremolo, fluctuating volume.
    - Surprise: Sudden sharp pitch leap, explosive attack, brief duration.
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    if emotion == "angry":
        f0 = np.random.uniform(220, 320)
        # Fast vibrato & harsh harmonics
        pitch_mod = f0 + 25 * np.sin(2 * np.pi * 6 * t)
        wave = (
            0.6 * np.sin(2 * np.pi * pitch_mod * t) +
            0.3 * np.sin(2 * np.pi * 2 * pitch_mod * t) +
            0.2 * np.sin(2 * np.pi * 3 * pitch_mod * t) +
            0.1 * np.random.normal(0, 0.05, len(t))
        )
        envelope = np.exp(-1.5 * (t - duration * 0.1)**2) # sudden punchy bursts
        wave = wave * (envelope + 0.4)
        
    elif emotion == "happy":
        f0 = np.random.uniform(200, 280)
        # Rising melodic contour with warm harmonics
        pitch_mod = f0 + 35 * np.sin(2 * np.pi * 3.5 * t) + 15 * np.sin(2 * np.pi * 7 * t)
        wave = (
            0.7 * np.sin(2 * np.pi * pitch_mod * t) +
            0.25 * np.sin(2 * np.pi * 2 * pitch_mod * t) +
            0.15 * np.sin(2 * np.pi * 3 * pitch_mod * t)
        )
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 2.0 * t))
        wave = wave * (envelope * 0.7 + 0.3)
        
    elif emotion == "sad":
        f0 = np.random.uniform(110, 160)
        # Downward pitch slope, quiet, subdued
        pitch_contour = f0 - 20 * (t / duration)
        wave = (
            0.8 * np.sin(2 * np.pi * pitch_contour * t) +
            0.15 * np.sin(2 * np.pi * 2 * pitch_contour * t)
        )
        envelope = np.exp(-0.8 * t) # gentle decay
        wave = wave * envelope * 0.5
        
    elif emotion == "fear":
        f0 = np.random.uniform(240, 340)
        # Rapid jitter and tremolo
        pitch_mod = f0 + 40 * np.sin(2 * np.pi * 12 * t) + 15 * np.random.normal(0, 5, len(t))
        tremolo = 0.6 + 0.4 * np.sin(2 * np.pi * 8 * t)
        wave = (
            0.65 * np.sin(2 * np.pi * pitch_mod * t) +
            0.2 * np.sin(2 * np.pi * 2 * pitch_mod * t) +
            0.15 * np.random.normal(0, 0.04, len(t))
        ) * tremolo
        
    elif emotion == "surprise":
        f0 = np.random.uniform(180, 240)
        # Sudden pitch jump in middle
        pitch_jump = f0 + 120 / (1 + np.exp(-15 * (t - duration * 0.3)))
        wave = (
            0.75 * np.sin(2 * np.pi * pitch_jump * t) +
            0.3 * np.sin(2 * np.pi * 2 * pitch_jump * t)
        )
        envelope = np.exp(-2.0 * np.abs(t - duration * 0.35))
        wave = wave * (envelope * 0.8 + 0.2)
        
    else: # neutral
        f0 = np.random.uniform(140, 180)
        # Steady pitch and constant calm cadence
        pitch_mod = f0 + 5 * np.sin(2 * np.pi * 1.5 * t)
        wave = (
            0.8 * np.sin(2 * np.pi * pitch_mod * t) +
            0.2 * np.sin(2 * np.pi * 2 * pitch_mod * t)
        )
        envelope = 0.6 * np.ones_like(t)
        wave = wave * envelope
        
    # Normalize to [-1.0, 1.0] and convert to 16-bit PCM
    wave = wave / (np.max(np.abs(wave)) + 1e-6)
    return wave.astype(np.float32)

def generate_speech_dataset(output_dir: str = None, samples_per_emotion: int = 100, sr: int = 22050):
    """
    Generates a folder hierarchy of audio WAV files partitioned by emotion classes.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "audio_samples")
        
    os.makedirs(output_dir, exist_ok=True)
    manifest = []
    
    print(f"[Task 2] Generating {len(EMOTIONS) * samples_per_emotion} audio samples across {len(EMOTIONS)} emotions...")
    sample_idx = 1
    
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i in range(samples_per_emotion):
            seed = i * 100 + hash(emotion) % 10000
            wave = synthesize_emotional_wave(emotion, duration=2.5, sr=sr, seed=seed)
            
            # Save WAV
            file_name = f"{emotion}_{i+1:03d}.wav"
            file_path = os.path.join(emotion_dir, file_name)
            
            # Convert float32 to int16 for WAV file
            int_wave = (wave * 32767).astype(np.int16)
            wavfile.write(file_path, sr, int_wave)
            
            manifest.append({
                "sample_id": sample_idx,
                "file_path": file_path,
                "relative_path": os.path.join(emotion, file_name),
                "emotion": emotion,
                "sample_rate": sr,
                "duration_seconds": 2.5
            })
            sample_idx += 1
            
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(os.path.dirname(output_dir), "dataset_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"[Task 2] Generated audio files saved to: {output_dir}")
    print(f"[Task 2] Dataset manifest saved to: {manifest_path}")
    return manifest_df

if __name__ == "__main__":
    generate_speech_dataset(samples_per_emotion=100)
