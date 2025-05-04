import torch
import torch.nn as nn

class FourierCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # [B, 32, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, H/2, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2)  # [B, 64, H/4, W/4]
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 64 * 64, 128),  # For 256x256 input → 64x64 after pooling
            nn.ReLU(),
            nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        return self.classifier(x)

# Optional: Add a pretrained weights loader
def load_pretrained(model, path="models/weights.pth"):
    model.load_state_dict(torch.load(path))
    return model