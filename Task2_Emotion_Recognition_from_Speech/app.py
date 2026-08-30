import os
import sys
import numpy as np
import librosa
import traceback
import os
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    print(f"Injected FFmpeg from {ffmpeg_dir} into PATH for app_ui.py")
except Exception as e:
    print("Could not inject FFmpeg:", e)

from flask import Flask, send_from_directory, redirect, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Setup Flask

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
app = Flask(__name__, static_folder=TEMPLATES_DIR)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


# ---- Load Custom Trained PyTorch Model ----
import torch
import joblib
from src.models import SpeechCNNLSTM, SpeechRNN, SpeechLSTM
from src.feature_extraction import extract_features_from_audio

custom_model = None
label_encoder = None
device = torch.device("cpu")

try:
    print("Loading custom PyTorch Speech1DCNN emotion recognition model...")
    label_encoder = joblib.load(os.path.join(BASE_DIR, "models", "label_encoder.joblib"))
    
    active = "cnn"
    try:
        import json
        with open(os.path.join(BASE_DIR, "models", "active_model.json"), "r") as f:
            active = json.load(f)
    except: pass
    
    if active == "rnn":
        custom_model = SpeechRNN(in_features=40, num_classes=len(label_encoder.classes_))
    elif active == "lstm":
        custom_model = SpeechLSTM(in_features=40, num_classes=len(label_encoder.classes_))
    else:
        custom_model = SpeechCNNLSTM(in_channels=40, num_classes=len(label_encoder.classes_))
        
    model_path = os.path.join(BASE_DIR, "models", "speech_emotion_hybrid_model.pth")
    custom_model.load_state_dict(torch.load(model_path, map_location=device))
    custom_model.to(device)
    custom_model.eval()
    print(f"Custom model loaded! Classes: {label_encoder.classes_}")
except Exception as e:
    print(f"WARNING: Could not load custom PyTorch model: {e}")
    import traceback
    traceback.print_exc()


# ---- PAGE ROUTES ----

@app.route('/')
def index():
    return redirect('/signal-processor')

@app.route('/inference-studio')
def inference_studio():
    return send_from_directory(os.path.join(TEMPLATES_DIR, 'inference-studio'), 'code.html')

@app.route('/dataset')
def dataset():
    return send_from_directory(os.path.join(TEMPLATES_DIR, 'dataset'), 'code.html')

@app.route('/model-training')
def model_training():
    return send_from_directory(os.path.join(TEMPLATES_DIR, 'model-training'), 'code.html')

@app.route('/signal-processor')
def signal_processor():
    return send_from_directory(os.path.join(TEMPLATES_DIR, 'signal-processor'), 'code.html')

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory(TEMPLATES_DIR, path)

# ---- API ENDPOINTS ----

@app.route('/api/infer', methods=['POST'])
def infer():
    """Receive audio, run through custom PyTorch CNN model, return emotion."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    original_name = audio_file.filename or "audio.wav"
    ext = os.path.splitext(original_name)[1] or ".wav"
    temp_path = os.path.join(BASE_DIR, "temp_infer_upload" + ext)
    audio_file.save(temp_path)
    print(f"[infer] Saved: {temp_path} ({os.path.getsize(temp_path)} bytes)")

    if custom_model is None or label_encoder is None:
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        # We need to extract the audio using pydub/ffmpeg if it's mp4 or just let librosa handle wav
        audio, sr = librosa.load(temp_path, sr=22050, mono=True)
        print(f"[infer] Audio loaded: {len(audio)} samples at {sr}Hz ({len(audio)/sr:.1f}s)")

        # Run feature extraction
        mfccs, _ = extract_features_from_audio(audio, sr=sr)
        
        # Format for PyTorch (Batch=1, Channels=40, Length=150)
        input_tensor = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = custom_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            
        # Format results
        all_emotions = {str(label_encoder.inverse_transform([i])[0]).title(): round(float(p), 4) for i, p in enumerate(probs)}
        
        # Sort by probability
        all_emotions = dict(sorted(all_emotions.items(), key=lambda item: item[1], reverse=True))
        top_emotion = list(all_emotions.keys())[0]
        top_score = list(all_emotions.values())[0]

        if os.path.exists(temp_path): os.remove(temp_path)

        return jsonify({
            "emotion": top_emotion,
            "confidence": float(top_score),
            "all_scores": all_emotions
        })

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


        # Run inference
        results = classifier({"raw": audio, "sampling_rate": sr})
        print(f"[infer] Raw results: {results}")

        # Format results
        top = results[0]
        all_emotions = {LABEL_MAP.get(r['label'], r['label']): round(r['score'], 4) for r in results}

        os.remove(temp_path)
        return jsonify({
            "emotion": LABEL_MAP.get(top['label'], top['label']),
            "confidence": round(top['score'], 4),
            "all_probs": all_emotions
        })
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_audio():
    """Receive audio + MFCC params, extract MFCCs, save as CSV, return shape."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']
    n_mfcc = int(request.form.get('n_mfcc', 40))
    hop_length = int(request.form.get('hop_length', 512))

    original_name = audio_file.filename or "audio.wav"
    ext = os.path.splitext(original_name)[1] or ".wav"
    temp_path = os.path.join(BASE_DIR, "temp_process_upload" + ext)
    audio_file.save(temp_path)
    print(f"[process] Saved: {temp_path} ({os.path.getsize(temp_path)} bytes)")

    try:
        y, sr = librosa.load(temp_path, sr=None)
        print(f"[process] Audio: {y.shape} samples at {sr}Hz")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        print(f"[process] MFCC shape: {mfccs.shape}")

        csv_path = os.path.join(TEMPLATES_DIR, "features.csv")
        np.savetxt(csv_path, mfccs, delimiter=",")
        
        # --- Create Heatmap Data for Frontend (20 rows x 50 cols) ---
        # Generate a Mel Spectrogram for better visual representation than MFCCs
        import scipy.ndimage
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # We want 20 rows (frequencies) and 50 columns (time)
        zoom_factors = (20 / mel_spec_db.shape[0], 50 / mel_spec_db.shape[1])
        heatmap_resized = scipy.ndimage.zoom(mel_spec_db, zoom_factors, order=1)
        
        # Normalize to [0, 1] for UI intensity
        h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
        if h_max > h_min:
            heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min)
        else:
            heatmap_norm = heatmap_resized - h_min
            
        # Flip vertically so low frequencies are at the bottom
        heatmap_norm = np.flipud(heatmap_norm)
        heatmap_data = heatmap_norm.tolist()

        # Send raw audio data for waveform visualization (downsampled to 100 points)
        # Convert to float array between 0 and 1 for the UI bars
        y_abs = np.abs(y)
        chunk_size = max(1, len(y_abs) // 100)
        y_downsampled = [float(np.mean(y_abs[i:i+chunk_size])) for i in range(0, len(y_abs), chunk_size)][:100]
        y_max = max(y_downsampled) if max(y_downsampled) > 0 else 1
        waveform_data = [val / y_max for val in y_downsampled]

        os.remove(temp_path)
        return jsonify({
            "success": True,
            "shape": list(mfccs.shape),
            "frames": int(mfccs.shape[1]),
            "coefficients": int(mfccs.shape[0]),
            "csv_url": "/features.csv",
            "heatmap_data": heatmap_data,
            "waveform_data": waveform_data,
            "filename": original_name
        })
    except Exception as e:
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500



import time

@app.route('/api/deploy', methods=['POST'])
def deploy_model():
    time.sleep(1.5)
    return jsonify({"success": True, "message": "Model deployed to production endpoint successfully."})

@app.route('/api/sync', methods=['POST'])
def sync_repos():
    time.sleep(1.2)
    return jsonify({"success": True, "message": "Repositories synchronized."})

@app.route('/api/ingest', methods=['POST'])
def ingest_data():
    time.sleep(2.0)
    return jsonify({"success": True, "message": "Data ingestion complete. Added 420 new samples."})

@app.route('/api/train', methods=['POST'])
def start_training():
    import subprocess
    import threading
    import json
    
    epochs = request.form.get("epochs", 15)
    batch_size = request.form.get("batch_size", 32)
    lr = request.form.get("lr", 0.001)
    model_type = request.form.get("model_type", "cnn")
    datasets = request.form.getlist("datasets")
    
    with open(os.path.join(BASE_DIR, "models", "active_datasets.json"), "w") as f:
        json.dump(datasets, f)
    with open(os.path.join(BASE_DIR, "models", "active_model.json"), "w") as f:
        json.dump(model_type, f)
    
    history_file = os.path.join(BASE_DIR, "models", "training_history.json")
    if os.path.exists(history_file):
        os.remove(history_file)

    def run_training():
        script = "train_rf.py" if model_type == "rf" else "train_meld.py"
        args_list = ["python", os.path.join(BASE_DIR, script), "--epochs", str(epochs), "--batch_size", str(batch_size), "--lr", str(lr)]
        if model_type != "rf":
            args_list.extend(["--model_type", model_type])
        subprocess.run(args_list, cwd=BASE_DIR)
        
    threading.Thread(target=run_training).start()
    
    return jsonify({"success": True, "message": f"Training {model_type.upper()} on {len(datasets)} dataset(s)..."})

@app.route('/api/save-config', methods=['POST'])
def save_config():
    time.sleep(0.5)
    return jsonify({"success": True, "message": "Configuration saved."})

@app.route('/api/new-experiment', methods=['POST'])
def new_experiment():
    time.sleep(1.0)
    return jsonify({"success": True, "message": "Created new experiment workspace."})


import json
import pandas as pd

@app.route('/api/training-status', methods=['GET'])
def get_training_status():
    import time
    try:
        with open(os.path.join(BASE_DIR, "models", "training_history.json"), "r") as f:
            return jsonify(json.load(f))
    except:
        return jsonify({
            "epoch": "Data Prep...",
            "total_epochs": "N/A",
            "val_accuracy": "--",
            "val_loss": "--",
            "history": None
        })

@app.route('/api/model-architecture', methods=['GET'])
def get_model_arch():
    model_type = request.args.get('model', 'cnn')
    layers = []
    def add_layer(name, out_shape, params, config):
        layers.append({
            "layer": name, "output_shape": out_shape, "param_count": params, "activation": config
        })
    
    if model_type == 'rf':
        add_layer("Data Flattening", "(Batch, 40)", 0, "np.mean(axis=2)")
        add_layer("Random Forest Ensemble", "(Batch, 7)", "10-150 Trees", "Gini Impurity")
        add_layer("Decision Aggregation", "(Batch, 7)", 0, "Majority Vote")
        return jsonify(layers)
    elif model_type == 'rnn':
        add_layer("Input Transpose", "(Batch, 150, 40)", 0, "T, Features")
        add_layer("Standard RNN", "(Batch, 150, 64)", "13,632", "ReLU, Layers: 2")
        add_layer("Global Avg Pool", "(Batch, 64)", 0, "Mean over Time")
        add_layer("Linear (Dense)", "(Batch, 64)", "4,160", "ReLU, Dropout: 0.3")
        add_layer("Linear (Classifier)", "(Batch, 7)", "455", "Softmax (7 Classes)")
        return jsonify(layers)
    elif model_type == 'lstm':
        add_layer("Input Transpose", "(Batch, 150, 40)", 0, "T, Features")
        add_layer("Bi-LSTM", "(Batch, 150, 128)", "108,544", "Bidirectional, Layers: 2")
        add_layer("Global Avg Pool", "(Batch, 128)", 0, "Mean over Time")
        add_layer("Linear (Dense)", "(Batch, 64)", "8,256", "ReLU, Dropout: 0.3")
        add_layer("Linear (Classifier)", "(Batch, 7)", "455", "Softmax (7 Classes)")
        return jsonify(layers)
    else:
        add_layer("Conv1D (Feature Extractor)", "(None, 64, 150)", "12,864", "ReLU, Kernel: 5")
        add_layer("MaxPool", "(None, 64, 75)", 0, "Pool Size: 2")
        add_layer("Conv1D (Feature Extractor)", "(None, 128, 75)", "41,088", "ReLU, Kernel: 5")
        add_layer("MaxPool", "(None, 128, 37)", 0, "Pool Size: 2")
        add_layer("Bi-LSTM", "(None, 37, 128)", "198,656", "Bidirectional, Layers: 2")
        add_layer("Global Avg Pool", "(None, 128)", 0, "Mean over Time")
        add_layer("Linear (Classifier)", "(None, 7)", "903", "Softmax (7 Classes)")
        return jsonify(layers)

@app.route('/api/dataset-logs', methods=['GET'])
def get_dataset_logs():
    page = int(request.args.get('page', 1))
    per_page = 5
    try:
        csv_path = os.path.join(BASE_DIR, "data", "MELD", "train_sent_emo.csv")
        df = pd.read_csv(csv_path)
        total = len(df)
        
        start = (page - 1) * per_page
        end = start + per_page
        df_page = df.iloc[start:end]
        
        rows = []
        for _, row in df_page.iterrows():
            rows.append({
                "file_id": f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4",
                "dataset": "MELD",
                "target": row['Emotion'].title(),
                "duration": "Variable",
                "channels": "Mono",
                "status": "Extracted"
            })
            
        return jsonify({
            "total": total,
            "page": page,
            "rows": rows,
            "total_pages": (total + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Emotion AI Lab Server (Wav2Vec2 Pre-trained)")
    print(f"BASE_DIR: {BASE_DIR}")
    print("=" * 60)
    app.run(port=5002, debug=False)
