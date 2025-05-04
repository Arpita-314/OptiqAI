import numpy as np
from skimage import transform
from PIL import Image

class OpticalPreprocessor:
    """Class for preprocessing optical images"""
    
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
    
    def run(self, image_paths):
        """Process a list of images"""
        processed = []
        for path in image_paths:
            img = self._load_and_preprocess(path)
            processed.append(img)
        return np.stack(processed)
    
    def _load_and_preprocess(self, path):
        """Load and preprocess a single image"""
        # Load image
        with Image.open(path) as img:
            if img.mode != 'L':
                img = img.convert('L')
            img = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img = img / np.max(img)
        
        # Resize if needed
        if img.shape != self.target_size:
            img = transform.resize(
                img,
                self.target_size,
                mode='constant',
                anti_aliasing=True
            )
        
        # Apply additional preprocessing steps
        img = self._enhance_contrast(img)
        img = self._remove_background(img)
        
        return img
    
    def _enhance_contrast(self, img):
        """Enhance image contrast"""
        p2, p98 = np.percentile(img, (2, 98))
        img_rescale = (img - p2) / (p98 - p2)
        return np.clip(img_rescale, 0, 1)
    
    def _remove_background(self, img):
        """Remove background noise"""
        # Simple threshold-based background removal
        threshold = np.mean(img) + 0.5 * np.std(img)
        img[img < threshold] = 0
        return img