import torch
from pathlib import Path

tensor_dir = Path(__file__).parent.parent.parent / "tensors"
tensor_dir.mkdir(parents=True, exist_ok=True)


def save_tensor(tensor: torch.Tensor, name: str):
    torch.save(tensor, tensor_dir / name)
