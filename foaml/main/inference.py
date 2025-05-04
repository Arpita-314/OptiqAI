import torch
from models.cnn import FourierCNN
from utils.data_loader import FourierDataset
import argparse

def predict(image_path, model_path="../models/checkpoint_epoch9.pth"):
    # Load model
    model = FourierCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and preprocess image
    dataset = FourierDataset("dummy_root")  # Hack to reuse transforms
    img = imread(image_path, as_gray=True).astype(np.float32)
    img = torch.tensor(np.expand_dims(img, axis=0))  # [1, H, W]
    
    # Predict
    with torch.no_grad():
        output = model(img.unsqueeze(0))  # Add batch dim
        _, predicted = torch.max(output, 1)
    
    class_names = ["single_slit", "double_slit"]  # Update with your classes
    return class_names[predicted.item()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    print(f"Prediction: {predict(args.image)}")