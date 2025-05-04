# run_mvp.py

import os
import torch
from torch.utils.data import random_split, DataLoader
from foaml.utils.data_loader import FourierDataset
from foaml.models.cnn import FourierCNN
from foaml.training.trainer import OpticsTrainer

def main():
    print("="*60)
    print("Welcome to the Fourier Optics AutoML CLI MVP!")
    print("This tool helps you preprocess your data and train a neural network model.")
    print("="*60)

    # 1. Get user input for data and parameters
    while True:
        data_dir = input("Enter the path to your data directory: ").strip()
        if os.path.isdir(data_dir):
            break
        print("Directory not found. Please try again.")

    try:
        pixel_size = float(input("Enter pixel size in micrometers (e.g., 5): "))
        wavelength = float(input("Enter wavelength in nanometers (e.g., 632.8): "))
    except ValueError:
        print("Invalid input for pixel size or wavelength. Exiting.")
        return

    # 2. Load data
    try:
        dataset = FourierDataset(data_dir)
        print(f"Loaded {len(dataset)} samples from {data_dir}. Classes: {dataset.classes}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if len(dataset) < 2:
        print("Not enough data to train. Please provide more samples.")
        return

    # 3. Split into train/val
    val_split = 0.2
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    # 4. DataLoaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Model
    model = FourierCNN(in_channels=1, num_classes=len(dataset.classes))

    # 6. Trainer
    trainer = OpticsTrainer(model, wavelength=wavelength, pixel_size=pixel_size)

    # 7. Training config
    config = {
        "epochs": 10,
        "patience": 3
    }

    # 8. Train the model
    print("\nStarting training...")
    trainer.manual_train(train_loader, val_loader, config)

    # 9. Simple result reporting
    print("\nEvaluating on validation set...")
    val_loss, phase_rmse = trainer.validate(val_loader)
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Phase RMSE: {phase_rmse:.4f}")

    print("\nMVP complete! You can now show this workflow to your users.")

if __name__ == "__main__":
    main()



