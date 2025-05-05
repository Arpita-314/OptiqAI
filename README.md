# Fourier Optics AutoML Framework

## Overview

The Fourier Optics AutoML Framework is a comprehensive, automated system for analyzing and processing optical data using both Fourier optics principles and machine learning techniques. This framework is designed to bridge the gap between traditional optical analysis and modern machine learning approaches, providing a powerful tool for researchers and engineers in the field of optics. It is still under development.

## Features

- **Data Ingestion**: Support for complex fields, intensity data, PSFs, and wavefront sensor measurements.
- **Preprocessing**: Includes DC removal, windowing, FFT operations, and phase unwrapping.
- **Model Selection**: Automated selection of physics-informed architectures (e.g., Fourier Neural Operators, U-Net variants).
- **Training**: Automates training with physics-aware loss functions and hyperparameter optimization.
- **Validation**: Provides Fourier optics-specific metrics like Strehl ratio, MTF correlation, RMS error, and phase RMSE.
- **Deployment**: Exports models to TorchScript and ONNX formats for cross-platform compatibility.
- **Visualization**: Plots intensity maps, phase maps, MTFs, and compares wavefront reconstructions.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for large-scale simulations)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fourier-optics-automl.git
   cd fourier-optics-automl
   ```

2. Create a virtual environment (optional, but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Quick Start

1. Run the main script:
   ```
   python main.py
   ```

2. Follow the prompts to input your data type, pixel size, and wavelength.

3. The framework will guide you through data preprocessing, model selection, training, and validation.

## Module Overview

- `data/`: Handles data ingestion and initial processing.
- `preprocessing/`: Implements Fourier optics-specific preprocessing techniques.
- `models/`: Contains model architectures and selection logic.
- `training/`: Manages the training process, including loss functions and optimization.
- `utils/`: Includes metrics, visualization tools, and deployment utilities.

## Advanced Usage

### Custom Model Integration

To add a custom model:

1. Create a new model class in `models/custom_models.py`.
2. Add the model to the selection options in `models/architecture.py`.

### Hyperparameter Tuning

The framework uses Optuna for hyperparameter optimization. Modify `training/trainer.py` to adjust the hyperparameter search space.

## Contributing

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments



## Contact

For questions or support, please open an issue on GitHub or contact paularpita.ap12@gmail.com.
