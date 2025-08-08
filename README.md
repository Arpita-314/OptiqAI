# FOAML: Fourier Optics AutoML Framework

A modular, production-ready framework for training and optimizing physics-informed neural networks in Fourier optics.

## Features

- Physics-informed loss functions for complex field reconstruction
- Early stopping and Optuna-based hyperparameter optimization
- Modular trainer class with model save/load utilities
- Device-agnostic (CPU/GPU) training
- Progress bars and logging for robust monitoring

## Installation

Clone the repository and install dependencies:
bash
pip install -r requirements.txt


## Usage

python
from trainer import OpticsTrainer
import torch

# Define your model (replace MyModel with your actual model class)
model = MyModel()

# Initialize trainer
trainer = OpticsTrainer(model, wavelength=632.8e-9, pixel_size=5e-6)

# Load data (replace with your actual data)
train_data = torch.randn(100, 1, 256, 256)
train_targets = torch.randn(100, 2, 256, 256)
val_data = torch.randn(20, 1, 256, 256)
val_targets = torch.randn(20, 2, 256, 256)

train_loader = trainer.create_dataloader(train_data, train_targets, batch_size=8)
val_loader = trainer.create_dataloader(val_data, val_targets, batch_size=8, shuffle=False)

# Manual training
config = {
    "epochs": 100,
    "patience": 10,
    "batch_size": 8,
    "auto_tune": False
}
trainer.manual_train(train_loader, val_loader, config)

# Save model
trainer.save_model("best_model.pth")


## Configuration

You can use a `config.yaml` file for training parameters:

yaml
epochs: 100
patience: 10
batch_size: 8
auto_tune: false


## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

MIT License

---

Contact:  
Your Name – paularpita.ap12@gmail.com

