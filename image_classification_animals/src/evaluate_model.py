import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model  # type: ignore


def evaluate_model(model_path, test_generator):
    """Evaluate the trained model on test data"""
    # Load the trained model
    model = load_model(model_path)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate metrics
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Print metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return y_true, y_pred, class_names, report

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save confusion matrix"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nConfusion matrix saved to 'results/confusion_matrix.png'")

if __name__ == "__main__":
    # This section will be called from main.py
    pass