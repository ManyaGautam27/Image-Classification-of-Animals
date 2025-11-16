import argparse
import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import verify_dataset, split_dataset, create_data_generators
from src.train_model import create_model, compile_and_train_model, plot_training_history
from src.evaluate_model import evaluate_model, plot_confusion_matrix
from src.predict_image import predict_image, preprocess_image
from src.utils import save_class_names, load_class_names, count_images_in_directory


def prepare_data():
    """Prepare and preprocess the data"""
    print("=== PREPARING DATA ===")
    
    # Verify dataset
    data_dir = "data/raw"
    class_names, class_counts = verify_dataset(data_dir)
    
    # Save class names for later use
    save_class_names(class_names)
    
    # Split dataset
    split_dataset(
        source_dir="data/raw",
        train_dir="data/train",
        val_dir="data/validation",
        test_dir="data/test"
    )
    
    # Count images in each set
    train_count = count_images_in_directory("data/train")
    val_count = count_images_in_directory("data/validation")
    test_count = count_images_in_directory("data/test")
    
    print(f"\nDataset Summary:")
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Test images: {test_count}")

def train_model():
    """Train the model"""
    print("=== TRAINING MODEL ===")
    
    # Create data generators
    train_gen, val_gen, _ = create_data_generators(
        train_dir="data/train",
        val_dir="data/validation",
        test_dir="data/test"
    )
    
    # Load class names
    class_names = load_class_names()
    num_classes = len(class_names)
    
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {', '.join(class_names)}")
    
    # Create model
    model, base_model = create_model(num_classes)
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    history = compile_and_train_model(model, train_gen, val_gen)
    
    # Plot training history
    plot_training_history(history)
    
    print("\nModel training completed!")
    print("Model saved to 'models/animal_classifier.keras'")

def evaluate():
    """Evaluate the trained model"""
    print("=== EVALUATING MODEL ===")
    
    # Load class names
    class_names = load_class_names()
    
    # Create test generator
    _, _, test_gen = create_data_generators(
        train_dir="data/train",
        val_dir="data/validation",
        test_dir="data/test"
    )
    
    # Evaluate model
    y_true, y_pred, class_names, report = evaluate_model(
        model_path='models/animal_classifier.keras',
        test_generator=test_gen
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    print("\nModel evaluation completed!")

def predict(image_path):
    """Predict a single image"""
    print("=== PREDICTING IMAGE ===")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    # Load class names
    class_names = load_class_names()
    
    # Predict image
    try:
        predicted_class, confidence = predict_image(
            model_path='models/animal_classifier.keras',
            img_path=image_path,
            class_names=class_names
        )
        print(f"\nPrediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Animal Classification Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare command
    subparsers.add_parser('prepare', help='Prepare and preprocess data')
    
    # Train command
    subparsers.add_parser('train', help='Train the model')
    
    # Evaluate command
    subparsers.add_parser('evaluate', help='Evaluate the trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict a single image')
    predict_parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_data()
    elif args.command == 'train':
        train_model()
    elif args.command == 'evaluate':
        evaluate()
    elif args.command == 'predict':
        predict(args.image_path)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()