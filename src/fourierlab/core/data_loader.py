import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FourierDataset(Dataset):
    """Dataset class for loading and managing optical data"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = []
        self.targets = []
        self.classes = []
        
        # Scan directory for images
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_paths.append(os.path.join(root, file))
                    # For now, use directory name as class
                    class_name = os.path.basename(root)
                    if class_name not in self.classes:
                        self.classes.append(class_name)
                    self.targets.append(self.classes.index(class_name))
        
        if not self.image_paths:
            raise ValueError(f"No valid images found in {data_dir}")
        
        # Get image shape from first image
        sample_img = self._load_image(self.image_paths[0])
        self.image_shape = sample_img.shape
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        target = self.targets[idx]
        return img, target
    
    def _load_image(self, path):
        """Load and preprocess a single image"""
        with Image.open(path) as img:
            # Convert to grayscale
            if img.mode != 'L':
                img = img.convert('L')
            # Convert to numpy array
            img = np.array(img, dtype=np.float32) / 255.0
            return img 