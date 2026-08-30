import os
import sys
import base64
import io
import json
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from predict import load_character_model, predict_word_or_sequence

app = Flask(__name__, template_folder='templates', static_folder='static')

try:
    global_model, global_classes = load_character_model()
    print("Model loaded successfully!")
except Exception as e:
    global_model, global_classes = None, None
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('testing/code.html')

@app.route('/architecture')
def architecture():
    return render_template('architecture/code.html')

@app.route('/datasets')
def datasets():
    return render_template('datasets/code.html')

@app.route('/logs')
def logs():
    return render_template('logs/code.html')


import subprocess

@app.route('/api/train', methods=['POST'])
def train_endpoint():
    with open('logs.txt', 'w') as f:
        f.write("Initializing Training Job...\n")
    subprocess.Popen(['python', '-u', 'train.py'], stdout=open('logs.txt', 'a'), stderr=subprocess.STDOUT)
    return jsonify({"success": True})

@app.route('/api/logs')
def logs_endpoint():
    if os.path.exists('logs.txt'):
        with open('logs.txt', 'r') as f:
            content = f.read()
            # replace carriage returns to make it nice
            content = content.replace('\r', '\n')
            return jsonify({"logs": content})
    return jsonify({"logs": "No logs found."})


import shutil
@app.route('/api/new_model', methods=['POST'])
def new_model_endpoint():
    # Reset logs
    if os.path.exists('logs.txt'):
        with open('logs.txt', 'w') as f:
            f.write("System Ready. Waiting for job...\n")
            
    # Reset model changes
    bak_path = os.path.join(base_dir, 'models', 'handwritten_crnn_best.pth.bak')
    curr_path = os.path.join(base_dir, 'models', 'handwritten_crnn_best.pth')
    if os.path.exists(bak_path):
        shutil.copy(bak_path, curr_path)
        
    global global_model
    try:
        global_model, _ = load_character_model()
    except:
        pass
        
    return jsonify({"success": True})

from flask import send_file
@app.route('/api/export_logs')
def export_logs():
    if os.path.exists('logs.txt'):
        return send_file('logs.txt', as_attachment=True)
    return "No logs found.", 404


@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    try:
        if not global_model:
            return jsonify({"success": False, "error": "Model not loaded. Please train first."}), 500
            
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400
            
        base64_img = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        res = predict_word_or_sequence(img, model=global_model, classes=global_classes)
        
        # Prepare image payload
        processed_arr = res.pop("processed_image")
        # Ensure values are mapped properly to 0-255 for display
        img_uint8 = np.clip(processed_arr * 255, 0, 255).astype(np.uint8)
            
        processed_pil = Image.fromarray(img_uint8, mode='L')
        buf = io.BytesIO()
        processed_pil.save(buf, format='PNG')
        res["processed_image_base64"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify(res)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5003, debug=True)
