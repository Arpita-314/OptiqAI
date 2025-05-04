from setuptools import setup, find_packages

setup(
    name="fourierlab",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PyQt5>=5.15.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "torch>=1.7.0",
        "scikit-image>=0.17.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-qt>=3.3.0",
        ],
    },
    python_requires=">=3.8",
) 