from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import sys
from pathlib import Path

class RustExtension(build_ext):
    """Custom build command for Rust extensions."""
    def run(self):
        # Build the Rust extension
        rust_dir = Path("src/fourierlab/core/rust")
        subprocess.run(
            [sys.executable, "build.py"],
            cwd=rust_dir,
            check=True,
        )
        super().run()

setup(
    name="fourierlab",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "PyQt5>=5.15.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "numba>=0.55.0",
        "mkl>=2023.0.0",  # Intel Math Kernel Library for CPU optimization
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x>=12.0.0",  # Uncomment if you have an NVIDIA GPU
            "pycuda>=2022.0.0",      # Uncomment if you have an NVIDIA GPU
        ],
    },
    python_requires=">=3.8",
    cmdclass={
        "build_ext": RustExtension,
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for Fourier optics simulations and inverse design",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fourierlab",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
) 