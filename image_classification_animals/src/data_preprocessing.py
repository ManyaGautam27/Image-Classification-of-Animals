import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import cv2


def verify_dataset(data_dir):
    """Verify the dataset and print summary"""
    print("Verifying dataset...")
    class_names = sorted(os.listdir(data_dir))
    class_names = [name for name in class_names if not name.startswith('.')]
    
    total_images = 0
    class_counts = {}
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            class_counts[class_name] = count
            total_images += count
            print(f"{class_name}: {count} images")
    
    print(f"\nTotal images: {total_images}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {', '.join(class_names)}")
    
    return class_names, class_counts

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train, validation, and test sets"""
    print("\nSplitting dataset...")
    
    # Create directories if they don't exist
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    class_names = sorted(os.listdir(source_dir))
    class_names = [name for name in class_names if not name.startswith('.')]
    
    total_train, total_val, total_test = 0, 0, 0
    
    for class_name in class_names:
        class_source = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_source):
            continue
            
        # Create class directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Get all images
        images = [f for f in os.listdir(class_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split images
        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
        
        # Copy images to respective directories
        for img in train_imgs:
            shutil.copy(os.path.join(class_source, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_source, img), os.path.join(val_dir, class_name, img))
        for img in test_imgs:
            shutil.copy(os.path.join(class_source, img), os.path.join(test_dir, class_name, img))
            
        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)
        
        print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    print(f"\nDataset split completed:")
    print(f"Train: {total_train} images")
    print(f"Validation: {total_val} images")
    print(f"Test: {total_test} images")

def create_data_generators(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """Create data generators with augmentation for training and preprocessing for validation/test"""
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    # Validation and test data generator (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator