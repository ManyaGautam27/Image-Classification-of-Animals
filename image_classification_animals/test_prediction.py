import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict_image import predict_image
from src.utils import load_class_names

def test_prediction():
    # Load class names
    class_names = load_class_names()
    print(f"Loaded class names: {class_names}")
    
    # Test prediction
    image_path = 'data/raw/Bear/Bear_10.jpg'
    if os.path.exists(image_path):
        print(f"Predicting image: {image_path}")
        predicted_class, confidence = predict_image(
            model_path='models/animal_classifier.keras',
            img_path=image_path,
            class_names=class_names
        )
        print(f"Prediction completed!")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence}")
    else:
        print(f"Image not found: {image_path}")

if __name__ == "__main__":
    test_prediction()