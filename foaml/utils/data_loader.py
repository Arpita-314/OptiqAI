from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
import numpy as np
import os

class FourierDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = sorted(os.listdir(root_dir))  # e.g., ["single_slit", "double_slit"]
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = imread(self.image_paths[idx], as_gray=True).astype(np.float32)
        img = np.expand_dims(img, axis=0)  # Add channel dim [1, H, W]
        if self.transform:
            img = self.transform(img)
        return torch.tensor(img), torch.tensor(self.labels[idx])

# Example transform (add to your pipeline if needed)
def default_transform(x):
    return (x - 0.5) / 0.5  # Simple normalization