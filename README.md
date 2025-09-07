# OptiqAI

<<<<<<< HEAD
A modular, production-ready framework for training and optimizing physics-informed neural networks in Fourier optics.
=======
## Overview

OptiqAI is a comprehensive, automated system for analyzing and processing optical data using both Fourier optics principles and machine learning techniques. This framework is designed to bridge the gap between traditional optical analysis and modern machine learning approaches, providing a powerful tool for researchers and engineers in the field of optics. It is still under development.
>>>>>>> cf27216d6a4a7908b9ffba1a1ef99bca08681347

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

<<<<<<< HEAD
Contributions are welcome! Please open issues or submit pull requests.
=======
I welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features.

## Citing This Work

If you use this framework in your research, please cite it as follows:

```
@software{fourier_optics_automl,
  author = {Arpita Paul},
  title = {Fourier Optics AutoML Framework},
  year = {2025},
  url = {https://github.com/yourusername/fourier-optics-automl}
}
```
>>>>>>> cf27216d6a4a7908b9ffba1a1ef99bca08681347

## License

MIT License

---

Contact:  
Your Name – paularpita.ap12@gmail.com

