import torch
import numpy as np


def to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """
    Converts a numpy array to a torch tensor on CPU/GPU.
    """
    if device == "cuda":
        return torch.Tensor(arr).cuda()
    elif device == "cpu":
        return torch.Tensor(arr)
    else:
        raise RuntimeError(f"Device type {device} unkown.")


def to_numpy(t: torch.Tensor, device: str) -> np.ndarray:
    """
    Converts a torch tensor on CPU/GPU to a numpy array
    """

    if device == "cuda":
        return t.cpu().detach().numpy()
    elif device == "cpu":
        return t.detach().numpy()
    else:
        raise RuntimeError(f"Device type {device} unkown.")
