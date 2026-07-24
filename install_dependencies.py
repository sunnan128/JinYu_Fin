#!/usr/bin/env python3
"""
Install all required dependencies for the Legal QA System
"""
import subprocess
import sys

required_packages = [
    "fastapi",
    "uvicorn",
    "python-multipart",
    "streamlit",
    "langchain",
    "langchain-community",
    "openai",
    "chromadb",
    "sentence-transformers",
    "rank-bm25",
    "pymupdf",
    "python-docx",
    "python-dotenv",
    "pydantic",
    "pydantic-settings",
    "jieba",
    "numpy",
    "pandas",
    "scikit-learn",
]

def install_package(package):
    """Install a single package using pip"""
    try:
        print(f"\nInstalling {package}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            package, "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def main():
    print("=" * 60)
    print("  Legal QA System - Dependency Installer")
    print("=" * 60)
    print()
    
    success_count = 0
    failed_packages = []
    
    for package in required_packages:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print()
    print("=" * 60)
    print(f"  Installation Summary")
    print("=" * 60)
    print(f"  Successfully installed: {success_count}/{len(required_packages)}")
    
    if failed_packages:
        print(f"  Failed packages: {', '.join(failed_packages)}")
        print("\nPlease try installing the failed packages manually:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
    else:
        print("\n  All dependencies installed successfully!")
    
    print()
    print("=" * 60)
    print("\nYou can now start the system!")
    print("\nStep 1 - Start Backend:")
    print("  python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload")
    print("\nStep 2 - Start Frontend (in new terminal):")
    print("  python -m streamlit run frontend/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false")

if __name__ == "__main__":
    main()
