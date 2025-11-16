import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore


def save_model(model, model_path):
    """Save the trained model"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

def load_trained_model(model_path):
    """Load a trained model"""
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training history"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_class_names(class_names, file_path='models/class_names.json'):
    """Save class names to a JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class names saved to {file_path}")

def load_class_names(file_path='models/class_names.json'):
    """Load class names from a JSON file"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    else:
        raise FileNotFoundError(f"Class names file not found at {file_path}")

def count_images_in_directory(directory):
    """Count total images in a directory"""
    total = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                total += 1
    return total

if __name__ == "__main__":
    # This section will be called from main.py
    pass