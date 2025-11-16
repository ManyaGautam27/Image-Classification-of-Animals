import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore
import cv2
from PIL import Image


def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess a single image for prediction"""
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values
    img_array /= 255.0
    
    return img_array

def predict_image(model_path, img_path, class_names):
    """Predict the class of a single image"""
    # Load the trained model
    model = load_model(model_path)
    
    # Preprocess the image
    processed_img = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    # Get class name
    predicted_class = class_names[predicted_class_idx]
    
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Print top 3 predictions
    top_3_idx = np.argsort(predictions[0])[::-1][:3]
    print(f"\nTop 3 Predictions:")
    for i, idx in enumerate(top_3_idx):
        class_name = class_names[idx]
        prob = predictions[0][idx]
        print(f"{i+1}. {class_name}: {prob:.4f}")
    
    return predicted_class, confidence

if __name__ == "__main__":
    # This section will be called from main.py
    pass