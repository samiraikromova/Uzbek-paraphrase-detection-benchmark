"""
Download the pretrained model weights ONCE into ./models/<name>.

This is the ONLY script in the project that uses the internet, and it only downloads public
model files from the Hugging Face Hub. It never reads or sends any data.
After this, the notebook runs fully offline.

Run from the project folder with the venv's Python (pip.exe and similar launchers are blocked):
    C:\\Users\\iutadmin\\venv\\Scripts\\python.exe download_models.py                (all models)
    C:\\Users\\iutadmin\\venv\\Scripts\\python.exe download_models.py mbert xlmr-base   (only some)
"""
import os
import sys

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

from huggingface_hub import list_repo_files, snapshot_download

# Local folder name -> Hugging Face repository (same names as MODELS in the notebook)
MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr-base": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "bge-m3": "BAAI/bge-m3",
    "tahrirchi-bert": "tahrirchi/tahrirchi-bert-base",
    "bertbek": "elmurod1202/bertbek-news-big-cased",
}

# Files we never need: TensorFlow / Flax / ONNX / Rust / Core ML / OpenVINO weights, images,
# and bge-m3's extra ColBERT/sparse heads (*.pt).
IGNORE = ["*.h5", "*.msgpack", "*.onnx", "*.onnx_data", "*.ot", "*.tflite", "*.mlmodel", "*.pt",
          "onnx/*", "openvino/*", "coreml/*", "flax_model*", "tf_model*", "rust_model*",
          "imgs/*", "*.png", "*.jpg"]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def folder_size_gb(path):
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1e9


def download(name):
    repo = MODELS[name]
    target = os.path.join(PROJECT_DIR, "models", name)
    ignore = list(IGNORE)
    # Many repos have the same weights twice (.safetensors and .bin): keep only safetensors.
    if any(f.endswith(".safetensors") for f in list_repo_files(repo)):
        ignore.append("*.bin")
    print(f"\n=== {name}  <-  {repo}")
    snapshot_download(repo_id=repo, local_dir=target, ignore_patterns=ignore)
    print(f"    saved to {target} ({folder_size_gb(target):.2f} GB)")


if __name__ == "__main__":
    names = sys.argv[1:] or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"Unknown model name(s): {unknown}. Choose from: {list(MODELS)}")
    failed = []
    for name in names:
        try:
            download(name)
        except Exception as e:  # keep going with the other models
            print(f"    FAILED: {type(e).__name__}: {e}")
            failed.append(name)
    print("\nDone." + (f" Failed: {failed}" if failed else " All downloads succeeded."))
