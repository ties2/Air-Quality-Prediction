#!/usr/bin/env python3
"""Setup script for nested_air_pollution package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "openaq>=0.3.0",
        "httpx>=0.25.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
    ]

setup(
    name="nested_air_pollution",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Nested Learning for Continual Air Quality Prediction using HOPE Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nested_air_pollution",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/nested_air_pollution/issues",
        "Documentation": "https://github.com/yourusername/nested_air_pollution#readme",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "ruff>=0.1.6",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
        "mlflow": [
            "mlflow>=2.8.0",
            "wandb>=0.16.0",
        ],
        "serve": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hope-train=scripts.train:main",
            "hope-fetch=scripts.fetch_data:main",
        ],
    },
)
