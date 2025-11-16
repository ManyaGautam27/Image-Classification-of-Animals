import os
import sys
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict_image import predict_image
from src.utils import load_class_names

def test_prediction():
    try:
        # Load class names
        class_names = load_class_names()
        
        # Test prediction
        image_path = 'data/raw/Bear/Bear_10.jpg'
        if os.path.exists(image_path):
            predicted_class, confidence = predict_image(
                model_path='models/animal_classifier.keras',
                img_path=image_path,
                class_names=class_names
            )
            
            # Write results to file
            result = {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "status": "success"
            }
            
            with open('prediction_result.json', 'w') as f:
                json.dump(result, f, indent=2)
                
            print("Prediction completed successfully!")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence}")
        else:
            print(f"Image not found: {image_path}")
    except Exception as e:
        error_result = {
            "error": str(e),
            "status": "error"
        }
        with open('prediction_result.json', 'w') as f:
            json.dump(error_result, f, indent=2)
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    test_prediction()