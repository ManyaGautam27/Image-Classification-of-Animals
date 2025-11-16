import os

def create_file(path):
    """Creates an empty file if it doesn't already exist."""
    with open(path, 'a', encoding='utf-8'):
        pass

def create_structure(base_dir="image_classification_animals"):
    # Define folder paths
    folders = [
        base_dir,
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "train"),
        os.path.join(base_dir, "data", "test"),
        os.path.join(base_dir, "data", "validation"),
        os.path.join(base_dir, "src"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "notebooks"),
    ]

    # Define file paths
    files = [
        os.path.join(base_dir, "main.py"),
        os.path.join(base_dir, "requirements.txt"),
        os.path.join(base_dir, "README.md"),
        os.path.join(base_dir, "src", "data_preprocessing.py"),
        os.path.join(base_dir, "src", "train_model.py"),
        os.path.join(base_dir, "src", "evaluate_model.py"),
        os.path.join(base_dir, "src", "predict_image.py"),
        os.path.join(base_dir, "src", "utils.py"),
    ]

    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Create files
    for file_path in files:
        create_file(file_path)

    print("✅ Project structure created successfully at:", os.path.abspath(base_dir))


if __name__ == "__main__":
    create_structure()
