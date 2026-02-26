from huggingface_hub import snapshot_download
from safetensors import safe_open
import os
import torch

path = snapshot_download(
    "janak22/index-tts-gpt",
    ignore_patterns=["*.tflite", "*.onnx", "*.msgpack", "*.ot", "*.h5"],
)

if os.path.exists(os.path.join(path, "model.safetensors")):
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
        for key in f.keys():
            print(key, f.get_tensor(key).dtype)
elif os.path.exists(os.path.join(path, "pytorch_model.bin")):
    file = torch.load(os.path.join(path, "pytorch_model.bin"))
    for name, param in file.items():
        print(name, param.dtype)
else:
    raise ValueError(f"Model file not found in {path}")
