"""
Task 2: Inference Engine for Speech Emotion Recognition
Predicts emotional state from any audio WAV file or audio signal array.
"""

import os
import sys
import joblib
import numpy as np
import torch
import torch.nn.functional as F

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.feature_extraction import extract_features_from_file, extract_features_from_audio
from src.models import Speech1DCNN

EMOTION_EMOJIS = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "neutral": "Neutral",
    "fear": "Fearful",
    "surprise": "Surprised"
}

def load_emotion_pipeline():
    models_dir = os.path.join(base_dir, "models")
    model_file = os.path.join(models_dir, "speech_emotion_best_model.pth")
    le_file = os.path.join(models_dir, "label_encoder.joblib")
    
    if not os.path.exists(model_file) or not os.path.exists(le_file):
        raise FileNotFoundError(f"Model artifacts not found in {models_dir}. Please run train.py first.")
        
    le = joblib.load(le_file)
    model = Speech1DCNN(in_channels=40, num_classes=len(le.classes_))
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    model.eval()
    return model, le

def predict_emotion(audio_input, sr: int = 22050, model=None, le=None) -> dict:
    """
    Accepts an audio file path (str) or a numpy waveform array (np.ndarray).
    Returns Top-1 predicted emotion, confidence score, and complete emotion probability distribution.
    """
    if model is None or le is None:
        model, le = load_emotion_pipeline()
        
    if isinstance(audio_input, str):
        mfcc_2d, _ = extract_features_from_file(audio_input, sr=sr, n_mfcc=40, max_len=120)
    elif isinstance(audio_input, np.ndarray):
        mfcc_2d, _ = extract_features_from_audio(audio_input, sr=sr, n_mfcc=40, max_len=120)
    else:
        raise ValueError("audio_input must be a file path string or numpy array.")
        
    # Input tensor shape: (1, 40, 120)
    input_tensor = torch.tensor(mfcc_2d, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()
        
    pred_idx = int(np.argmax(probs))
    pred_emotion = le.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    
    prob_dist = {str(le.classes_[i]): round(float(probs[i]), 4) for i in range(len(le.classes_))}
    
    return {
        "predicted_emotion": pred_emotion,
        "display_name": EMOTION_EMOJIS.get(pred_emotion, pred_emotion.capitalize()),
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.1f}%",
        "probabilities": prob_dist
    }

if __name__ == "__main__":
    print("=" * 65)
    print("      TASK 2: SPEECH EMOTION RECOGNITION INFERENCE")
    print("=" * 65)
    
    data_dir = os.path.join(base_dir, "data", "audio_samples")
    test_emotions = ["happy", "angry", "sad", "neutral", "fear", "surprise"]
    
    try:
        model, le = load_emotion_pipeline()
        print(f"Loaded Speech Emotion Model. Supported Emotions: {list(le.classes_)}\n")
        
        for emotion in test_emotions:
            sample_file = os.path.join(data_dir, emotion, f"{emotion}_001.wav")
            if os.path.exists(sample_file):
                result = predict_emotion(sample_file, model=model, le=le)
                print(f"File: [{emotion}_001.wav] (Actual: {emotion})")
                print(f"  -> Predicted: {result['display_name']} (Confidence: {result['confidence_percentage']})")
                print(f"  -> Probabilities: {result['probabilities']}\n")
    except Exception as e:
        print(f"Note: Model needs training first: {e}")
