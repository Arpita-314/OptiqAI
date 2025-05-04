train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cnn import FourierCNN
from utils.data_loader import FourierDataset

# Config
config = {
    "data_dir": "../data/train",
    "batch_size": 8,
    "lr": 0.001,
    "epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def train():
    # Data
    train_data = FourierDataset(config["data_dir"])
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)

    # Model
    model = FourierCNN().to(config["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    for epoch in range(config["epochs"]):
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(config["device"]), targets.to(config["device"])
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"../models/checkpoint_epoch{epoch}.pth")

if __name__ == "__main__":
    train()