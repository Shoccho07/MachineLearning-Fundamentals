from flask import Flask, request, render_template, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import joblib

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = set()  # Empty set = allow any extension

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.secret_key = 'replace_this_with_a_secure_random_key'

# --- Load model ---
MODEL_PATH = 'models/combined_model.pkl'
data = joblib.load(MODEL_PATH)
model = data['svm']
scaler = data['scaler']

# --- Helpers ---
def allowed_file(filename):
    return True  # accept all file extensions

def crop_and_preprocess(image):
    """Detect faces, return list of flattened arrays and bounding boxes"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Assuming model trained on RGB
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    processed_faces = []
    boxes = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))  # Resize to match training
        processed_faces.append(face.flatten())
        boxes.append((x, y, w, h))
    return processed_faces, boxes

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Read image
        img = cv2.imread(path)
        if img is None:
            flash("Cannot read image")
            return redirect(url_for('index'))

        faces, boxes = crop_and_preprocess(img)
        if not faces:
            flash("No faces detected")
            return redirect(url_for('index'))

        # Predict each face
        predictions = []
        for face in faces:
            try:
                face_scaled = scaler.transform([face])
                pred = model.predict(face_scaled)[0]
                predictions.append(pred)
            except Exception as e:
                predictions.append(f"Error: {str(e)}")

        # Draw boxes and labels
        for (x, y, w, h), label in zip(boxes, predictions):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save result image
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(result_path, img)

        flash(f"Predictions: {predictions}")
        return render_template('index.html', result_image=filename)
    else:
        flash("Invalid file")
        return redirect(url_for('index'))

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

# --- Run app ---
if __name__ == '__main__':
    app.run(debug=True)
