from flask import Flask, render_template, request, jsonify
import os
import model
from utils import extract_features

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        audio_files = request.files.getlist('files')
        labels = request.form.getlist('labels')

        # Save uploaded files and prepare data
        file_paths = []
        for i, file in enumerate(audio_files):
            save_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(save_path)
            file_paths.append((save_path, labels[i]))

        # Train the model
        model.train_model(file_paths)
        return jsonify({'message': 'Model trained successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process-command', methods=['POST'])
def process_command():
    try:
        audio = request.files['audio']
        save_path = os.path.join(UPLOAD_FOLDER, audio.filename)
        audio.save(save_path)

        # Perform inference
        predicted_command = model.predict_command(save_path)
        result = model.execute_command(predicted_command)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
