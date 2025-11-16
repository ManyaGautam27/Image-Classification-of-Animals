import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import json
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict_image import predict_image, preprocess_image
from src.utils import load_class_names


class AnimalClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Classifier")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.image_path = None
        self.class_names = []
        self.load_class_names()
        
        # Create UI
        self.create_widgets()
        
    def load_class_names(self):
        """Load class names from the model"""
        try:
            self.class_names = load_class_names()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load class names: {str(e)}")
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Animal Classifier", font=("Arial", 24, "bold"), 
                              bg='#f0f0f0', fg='#333333')
        title_label.pack(pady=(0, 20))
        
        # Instruction label
        instruction_label = tk.Label(main_frame, text="Select an animal image to classify", 
                                    font=("Arial", 12), bg='#f0f0f0', fg='#666666')
        instruction_label.pack(pady=(0, 20))
        
        # Browse button with better styling
        browse_frame = tk.Frame(main_frame, bg='#f0f0f0')
        browse_frame.pack(pady=(0, 20))
        
        self.browse_button = tk.Button(browse_frame, text="Browse Image", 
                                      command=self.browse_image,
                                      font=("Arial", 12, "bold"),
                                      bg='#4CAF50', fg='white',
                                      padx=20, pady=10,
                                      relief=tk.RAISED, bd=2,
                                      cursor="hand2")
        self.browse_button.pack()
        
        # Image path display
        self.image_path_label = tk.Label(browse_frame, text="No image selected", 
                                        font=("Arial", 10), bg='#f0f0f0', fg='#888888')
        self.image_path_label.pack(pady=(10, 0))
        
        # Content area (image and results)
        content_frame = tk.Frame(main_frame, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Image display
        left_panel = tk.LabelFrame(content_frame, text="Image Preview", 
                                  font=("Arial", 12, "bold"),
                                  bg='#f0f0f0', fg='#333333',
                                  padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas for image display
        self.image_canvas = tk.Canvas(left_panel, bg='white', relief=tk.SUNKEN, bd=1)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Results
        right_panel = tk.LabelFrame(content_frame, text="Classification Results", 
                                   font=("Arial", 12, "bold"),
                                   bg='#f0f0f0', fg='#333333',
                                   padx=10, pady=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Prediction result
        self.prediction_label = tk.Label(right_panel, text="Prediction: ", 
                                        font=("Arial", 14, "bold"),
                                        bg='#f0f0f0', fg='#333333')
        self.prediction_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.confidence_label = tk.Label(right_panel, text="Confidence: ", 
                                        font=("Arial", 12),
                                        bg='#f0f0f0', fg='#333333')
        self.confidence_label.pack(anchor=tk.W, pady=(0, 20))
        
        # Top predictions header
        top_pred_header = tk.Label(right_panel, text="Top Predictions:", 
                                  font=("Arial", 12, "bold"),
                                  bg='#f0f0f0', fg='#333333')
        top_pred_header.pack(anchor=tk.W, pady=(0, 10))
        
        # Top predictions listbox with scrollbar
        listbox_frame = tk.Frame(right_panel, bg='#f0f0f0')
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.top_predictions_listbox = tk.Listbox(listbox_frame, 
                                                 font=("Arial", 10),
                                                 bg='white', fg='#333333',
                                                 selectbackground='#4CAF50',
                                                 relief=tk.SUNKEN, bd=1)
        scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, 
                                command=self.top_predictions_listbox.yview)
        self.top_predictions_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.top_predictions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Classify button
        self.classify_button = tk.Button(main_frame, text="Classify Image", 
                                        command=self.classify_image,
                                        font=("Arial", 12, "bold"),
                                        bg='#2196F3', fg='white',
                                        padx=20, pady=10,
                                        relief=tk.RAISED, bd=2,
                                        state=tk.DISABLED,
                                        cursor="hand2")
        self.classify_button.pack(pady=(20, 0))
        
    def browse_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select an animal image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.image_path_label.config(text=os.path.basename(file_path))
            self.classify_button.config(state=tk.NORMAL)
            self.display_image(file_path)
            
    def display_image(self, image_path):
        """Display the selected image"""
        try:
            # Clear canvas
            self.image_canvas.delete("all")
            
            # Open and resize image
            image = Image.open(image_path)
            
            # Get canvas dimensions
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # If canvas size is not yet determined, use a default size
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 400
                canvas_height = 400
            
            # Resize image to fit canvas while maintaining aspect ratio
            image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)  # type: ignore
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Display image in center of canvas
            x = (canvas_width - image.width) // 2
            y = (canvas_height - image.height) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.image_canvas.image = photo  # Keep a reference # type: ignore
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            
    def classify_image(self):
        """Classify the selected image"""
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        if not self.class_names:
            messagebox.showerror("Error", "Class names not loaded. Please check your model.")
            return
            
        try:
            # Show loading message
            self.prediction_label.config(text="Classifying...")
            self.confidence_label.config(text="Please wait...")
            self.top_predictions_listbox.delete(0, tk.END)
            self.top_predictions_listbox.insert(tk.END, "Analyzing image...")
            self.root.update()
            
            # Perform prediction
            predicted_class, confidence = predict_image(
                model_path='models/animal_classifier.keras',
                img_path=self.image_path,
                class_names=self.class_names
            )
            
            # Update results
            self.prediction_label.config(text=f"Prediction: {predicted_class}")
            self.confidence_label.config(text=f"Confidence: {confidence:.4f}")
            
            # Get top predictions for detailed view
            processed_img = preprocess_image(self.image_path)
            
            # Load model for additional predictions
            model = load_model('models/animal_classifier.keras')
            predictions = model.predict(processed_img, verbose=0)
            
            top_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5 predictions
            
            # Update top predictions listbox
            self.top_predictions_listbox.delete(0, tk.END)
            
            for i, idx in enumerate(top_indices):
                class_name = self.class_names[idx]
                prob = predictions[0][idx]
                self.top_predictions_listbox.insert(tk.END, f"{i+1}. {class_name}: {prob:.4f}")
                
                # Highlight the top prediction
                if i == 0:
                    self.top_predictions_listbox.itemconfig(i, {'bg': '#E8F5E9'})
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during classification: {str(e)}")
            self.prediction_label.config(text="Prediction: Error")
            self.confidence_label.config(text="Confidence: N/A")
            self.top_predictions_listbox.delete(0, tk.END)
            self.top_predictions_listbox.insert(tk.END, "Classification failed")


def main():
    root = tk.Tk()
    app = AnimalClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()