# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:38:52 2023

@author: doguk
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pydub import AudioSegment
import librosa
from scipy.signal import filtfilt, butter
from scipy.signal import wiener

#path
targetPath = ".\Target"
currPath = ".\Current"

#User-defined parameters
n_epochs = int(input("Enter number of epochs: "))
print(torch.version.cuda)

def load_voice_samples(currPath, targetPath):
    current_voice_paths = [os.path.join(currPath, f) for f in os.listdir(currPath) if f.endswith('.mp3') or f.endswith('.wav')]
    target_voice_paths = [os.path.join(targetPath, f) for f in os.listdir(targetPath) if f.endswith('.mp3') or f.endswith('.wav')]
    current_voice_samples = [AudioSegment.from_file(path) for path in current_voice_paths]
    target_voice_samples = [AudioSegment.from_file(path) for path in target_voice_paths]
    return current_voice_samples, target_voice_samples

def denoise(audio):
    return wiener(audio)

def pitch_correction(audio, sr):
    pitches, _ = librosa.core.piptrack(y=audio, sr=sr)
    return pitches

def equalize(audio):
    b, a = butter(4, [0.1, 0.9], btype='band')
    return filtfilt(b, a, audio)

def normalize_volume(audio):
    return audio / np.max(np.abs(audio))

def preprocess(audio_samples):
    features = []
    for audio in audio_samples:
        # Convert PyDub audio to NumPy array
        audio_array = np.array(audio.get_array_of_samples())
        
        # Convert to floating-point
        audio_array = audio_array.astype(np.float32) / 32767.0  # Assuming 16-bit PCM
        
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=audio_array, sr=audio.frame_rate, n_mfcc=13)
        
        features.append(mfcc)
        
    return features

def post_process(audio, sample_rate=44100):
    # Denoising
    audio = denoise(audio)
    
    # Pitch Correction
    audio = pitch_correction(audio, sample_rate)
    
    # Volume Normalization
    audio = normalize_volume(audio)
    
    # Equalization
    audio = equalize(audio)
    
    return audio

def generate_audio(model, input_features, vocoder , sample_rate=44100):
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Add a batch dimension and ensure the data type is float32
    input_features = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        output_features = model(input_features)
    
    # Remove batch dimension for the vocoder
    output_features = output_features.squeeze(0)
    output_audio_array = post_process(output_features)
    output_audio_array = output_audio_array.astype(np.int16)
    print("Data type of output_audio_array:", output_audio_array.dtype)
    print("Sample width:", output_audio_array.dtype.itemsize)
    
    # Convert NumPy array to PyDub audio
    output_audio = AudioSegment(
        output_audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=output_audio_array.dtype.itemsize,
        channels=1
    )
    
    # Export as MP3 file
    output_audio.export("output_audio.mp3", format="mp3", bitrate="128k")

# Voice Cloning Model
class VoiceCloningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VoiceCloningModel, self).__init__()
        
        # Increased complexity with more layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        x = x.view(x.size(0), -1)  # Flatten the input    
        # Activation functions and dropout layers added
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x
    
def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Flatten and convert to PyTorch tensors
    X_train_flat = [x.flatten() for x in X_train]
    y_train_flat = [y.flatten() for y in y_train]
   
    X_train_tensor = torch.tensor(np.array(X_train_flat), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train_flat), dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item()}")
        
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
    
    y_tests = torch.tensor(y_test, dtype=torch.float32)
    output = torch.tensor(outputs, dtype=torch.float32)
    # Flatten or reshape
    y_test_flat = y_tests.view(-1)
    outputs_flat = output.view(-1)
    
    # Calculate MSE
    mse = mean_squared_error(y_test_flat.numpy(), outputs_flat.numpy())
    print(f"Mean Squared Error on Test Set: {mse}")
        
def main():
    # Data collection
    current_voice_samples, target_voice_samples = load_voice_samples(currPath, targetPath)
    print(f"Loaded {len(current_voice_samples)} current voice samples and {len(target_voice_samples)} target voice samples.")

    # Preprocessing
    X = preprocess(current_voice_samples)
    y = preprocess(target_voice_samples)
    print(f"Generated {len(X)} current voice features and {len(y)} target voice features.")


    # Initialize model
    input_dim = X[0].shape[0] * X[0].shape[1]  # Assuming X_train[0] is a 2D array
    output_dim = y[0].shape[0] * y[0].shape[1]  # Assuming y_train[0] is a 2D array
    
    model = VoiceCloningModel(input_dim, output_dim)

    # Train model
    train_model(model, X, y, n_epochs)

    # Evaluate model
    evaluate_model(model, X, y)
    waveglow = ""

    # Generate new audio
    sample_rate = current_voice_samples[0].frame_rate
    generate_audio(model, X[0], waveglow, sample_rate)
    
if __name__ == "__main__":
    main()