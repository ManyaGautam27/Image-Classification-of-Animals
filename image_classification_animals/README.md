# Animal Classification Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An image classification system that can identify and classify 15 different animal species using deep learning with transfer learning techniques.

## 🐶 Project Overview

This project implements an animal classification system capable of identifying 15 different animal species from images. It uses transfer learning with MobileNetV2 as the base model to achieve high accuracy with relatively fast training times.

### Supported Animals
- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

## 🚀 Features

- **Deep Learning Model**: Uses MobileNetV2 with transfer learning for efficient training
- **GUI Application**: User-friendly graphical interface for easy image classification
- **Command-line Interface**: Full pipeline control through CLI commands
- **Data Preprocessing**: Automatic dataset organization and augmentation
- **Model Evaluation**: Comprehensive model evaluation with confusion matrix
- **Visualization**: Training history plots and performance metrics

## 📁 Project Structure

```
image_classification_animals/
├── data/
│   ├── raw/           # Original dataset
│   ├── train/         # Training set (80% split)
│   ├── validation/    # Validation set (10% split)
│   └── test/          # Test set (10% split)
├── models/
│   ├── animal_classifier.keras  # Trained model
│   └── class_names.json         # Class labels
├── src/
│   ├── data_preprocessing.py    # Data handling functions
│   ├── train_model.py           # Model creation and training
│   ├── evaluate_model.py        # Model evaluation functions
│   ├── predict_image.py         # Image prediction functions
│   └── utils.py                 # Utility functions
├── results/
│   ├── training_history.png     # Training plots
│   └── confusion_matrix.png     # Confusion matrix visualization
├── main.py                      # Main CLI interface
├── gui_app.py                   # Graphical user interface
└── requirements.txt             # Project dependencies
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd image_classification_animals
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Command Line Interface

The project offers a full pipeline through command-line commands:

1. **Prepare Data**:
```bash
python main.py prepare
```

2. **Train Model**:
```bash
python main.py train
```

3. **Evaluate Model**:
```bash
python main.py evaluate
```

4. **Predict Single Image**:
```bash
python main.py predict --image_path path/to/your/image.jpg
```

### Graphical User Interface

Run the GUI application for an interactive experience:
```bash
python gui_app.py
```

In the GUI:
1. Click "Browse Image" to select an animal image
2. Click "Classify Image" to get the prediction
3. View the top predictions with confidence scores

## 🧠 Model Architecture

The model uses **MobileNetV2** as the base architecture with transfer learning:
- Pre-trained on ImageNet dataset
- Custom classification head with dropout for regularization
- Optimized for mobile and embedded vision applications
- Fast inference with good accuracy

## 📊 Performance

The model achieves high accuracy on the test set:
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~93%
- **Test Accuracy**: ~92%

Performance metrics are visualized in:
- `results/training_history.png`: Training and validation curves
- `results/confusion_matrix.png`: Class-wise performance

## 📦 Requirements

- Python 3.8+
- TensorFlow 2.14+
- Keras 2.14+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV-Python
- Pillow

See [requirements.txt](requirements.txt) for detailed versions.

## 🎯 Dataset

The project uses a custom dataset containing images of 15 animal species with approximately 130 images per class. The dataset is automatically split into:
- 80% for training
- 10% for validation
- 10% for testing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues and pull requests.

## 🙏 Acknowledgments

- Thanks to the TensorFlow team for the excellent deep learning framework
- Inspired by various computer vision projects in the community