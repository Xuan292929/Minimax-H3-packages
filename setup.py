# SPDX-License-Identifier: Apache-2.0
from setuptools import setup, find_packages

setup(
    name="packages",
    version="0.1.0",
    description="Industrial-grade pure PyTorch algorithm packages repository (featuring MiniMax-H3)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "torchaudio>=2.4.0",
        "transformers>=4.45.0",
        "safetensors>=0.4.0",
        "einops>=0.8.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "tqdm>=4.65.0",
    ],
)
