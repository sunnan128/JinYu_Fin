#!/usr/bin/env python3
"""
Download embedding model for JinYu Financial Intelligence System.
Supports multiple download sources for Chinese users.
"""
import os
import sys
import shutil

MODEL_CACHE_DIR = "./backend/data/model_cache"
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
MODEL_SAFE = MODEL_NAME.replace("/", "_")

# Set HF mirror for Chinese users
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath(MODEL_CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.abspath(MODEL_CACHE_DIR)

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

CACHED_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, MODEL_SAFE)


def download_from_hf_mirror():
    """Download model from Hugging Face mirror"""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"Downloading {MODEL_NAME} from HF mirror...")
        model = SentenceTransformer(MODEL_NAME)
        print(f"Model downloaded and cached successfully!")
        return True
    except Exception as e:
        print(f"HF mirror download failed: {e}")
        return False


def main():
    print("=" * 60)
    print("  JinYu Model Downloader")
    print(f"  Target: {MODEL_NAME}")
    print("=" * 60)
    print()
    print(f"Model cache directory: {os.path.abspath(MODEL_CACHE_DIR)}")
    print()

    # Clean corrupted cache
    if os.path.exists(CACHED_MODEL_PATH):
        print(f"Clearing existing cache: {CACHED_MODEL_PATH}")
        shutil.rmtree(CACHED_MODEL_PATH)

    # Remove any leftover HF hub locks
    locks_dir = os.path.join(MODEL_CACHE_DIR, ".locks")
    if os.path.exists(locks_dir):
        shutil.rmtree(locks_dir)

    print(f"\nDownloading {MODEL_NAME} from HF Mirror...")
    result = download_from_hf_mirror()
    if result:
        print("\nSuccess! Model downloaded and ready.")
        return

    print("\n" + "=" * 60)
    print("  Download failed!")
    print("=" * 60)
    print()
    print("Please check:")
    print("1. Network connection")
    print("2. HF mirror availability: https://hf-mirror.com")
    print(f"3. Or manually place model at: {os.path.abspath(CACHED_MODEL_PATH)}")
    print()

if __name__ == "__main__":
    main()
