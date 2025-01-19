from scipy.io.wavfile import read
from python_speech_features import mfcc
import numpy as np

def extract_features(audio_path):
    try:
        sample_rate, signal = read(audio_path)
        features = mfcc(signal, samplerate=sample_rate, numcep=13, nfft=512)
        return np.mean(features, axis=0)
    except Exception as e:
        raise ValueError(f"Error extracting features: {e}")
