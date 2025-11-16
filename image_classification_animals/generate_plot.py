import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train_model import plot_training_history

# Mock history data
class History:
    def __init__(self):
        self.history = {
            'accuracy': [0.6901, 0.8024],
            'val_accuracy': [0.8247, 0.8917],
            'loss': [1.0151, 0.7026],
            'val_loss': [0.5357, 0.3434]
        }

history = History()

# Generate plot
plot_training_history(history)
print("Training history plot generated!")