# FourierLab

A GPU-accelerated Fourier optics simulation and inverse design tool for photonics applications.

## Features

- GPU-accelerated wave propagation using NVIDIA CUDA
- Physics-based simulation with complex media support
- Inverse design optimization
- Interactive GUI for parameter control
- Pattern generation and analysis
- Dataset generation and management

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x
- PyTorch with CUDA support
- PyQt5
- NumPy
- Pillow
- CuPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fourierlab.git
cd fourierlab
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install CUDA Toolkit:
- Download and install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- Set environment variables:
  - CUDA_PATH: Path to CUDA installation
  - Add to PATH:
    - %CUDA_PATH%\bin
    - %CUDA_PATH%\libnvvp

## Usage

1. Run the GUI application:
```bash
python run_gui.py
```

2. The application will automatically detect CUDA availability and use GPU acceleration when available.

## Project Structure

```
fourierlab/
├── src/
│   ├── core/
│   │   ├── wave_propagator.py
│   │   ├── phase_mask.py
│   │   └── pattern_generator.py
│   ├── UI/
│   │   └── gui/
│   │       ├── main_window.py
│   │       ├── data_manager.py
│   │       └── inverse_design_manager.py
│   └── utils/
│       └── visualization.py
├── tests/
├── examples/
├── requirements.txt
└── setup.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
