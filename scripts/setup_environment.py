"""
Setup script for Flickd AI Engine dependencies
This script installs and configures all required packages for the ML pipeline
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")

def main():
    print("Setting up Flickd AI Engine environment...")
    
    # Core ML packages
    packages = [
        "ultralytics",  # YOLOv8
        "torch",        # PyTorch
        "torchvision",  # Computer Vision
        "transformers", # HuggingFace models
        "clip-by-openai", # CLIP embeddings
        "faiss-cpu",    # Vector similarity search
        "opencv-python", # Video processing
        "pillow",       # Image processing
        "numpy",        # Numerical computing
        "pandas",       # Data manipulation
        "scikit-learn", # ML utilities
        "fastapi",      # API framework
        "uvicorn",      # ASGI server
        "python-multipart", # File uploads
        "spacy",        # NLP processing
    ]
    
    print(f"Installing {len(packages)} packages...")
    
    for package in packages:
        install_package(package)
    
    # Download spaCy model
    print("\nDownloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully downloaded spaCy model")
    except subprocess.CalledProcessError:
        print("Failed to download spaCy model")
    
    print("\nEnvironment setup complete")
    print("\nNext steps:")
    print("1. Place your video files in the /data/videos/ directory")
    print("2. Add your product catalog CSV to /data/products.csv")
    print("3. Run the main processing script")

if __name__ == "__main__":
    main()
