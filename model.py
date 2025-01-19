import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import extract_features
import os

MODEL_PATH = 'voice_calculator_model.pkl'


def load_training_data(data_dir):
    """
    Load audio file paths and corresponding labels.
    Args:
        data_dir (str): Path to directory containing audio files organized in subdirectories for each label.
    
    Returns:
        list: List of (file_path, label) tuples.
    """
    file_paths = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for file_name in os.listdir(label_dir):
                if file_name.endswith('.wav'):
                    file_paths.append((os.path.join(label_dir, file_name), label))
    return file_paths


def prepare_features_labels(file_paths):
    """
    Extract features and labels from audio files.
    Args:
        file_paths (list): List of (file_path, label) tuples.

    Returns:
        np.ndarray: Extracted features.
        np.ndarray: Corresponding labels.
    """
    X, y = [], []
    for file_path, label in file_paths:
        try:
            features = extract_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return np.array(X), np.array(y)


def train_and_save_model(data_dir):
    """
    Train the model and save it to disk.
    Args:
        data_dir (str): Path to directory containing training data.
    """
    print("Loading training data...")
    file_paths = load_training_data(data_dir)

    print("Extracting features...")
    X, y = prepare_features_labels(file_paths)

    print(f"Feature matrix shape: {X.shape}, Labels: {len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training the model...")
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    print("Evaluating the model...")
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {accuracy:.2f}")

    print(f"Saving the model to {MODEL_PATH}...")
    joblib.dump(classifier, MODEL_PATH)
    print("Model saved successfully!")


def predict_command(audio_path):
    """
    Predict the label for a given audio file.
    Args:
        audio_path (str): Path to the audio file.

    Returns:
        str: Predicted label.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    print("Loading the model...")
    classifier = joblib.load(MODEL_PATH)

    print("Extracting features from audio...")
    features = extract_features(audio_path)

    print("Predicting the command...")
    prediction = classifier.predict([features])[0]
    return prediction


if __name__ == "__main__":
    data_directory = "uploads"  # Path to the training dataset
    train_and_save_model(data_directory)
