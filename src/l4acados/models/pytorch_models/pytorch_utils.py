import torch
import numpy as np
from time import perf_counter


def to_tensor(arr: np.ndarray, device: str, synchronize=False) -> torch.Tensor:
    """
    Converts a numpy array to a torch tensor on CPU/GPU.
    """
    time_start = perf_counter()
    if device == "cuda":
        result = torch.Tensor(arr).cuda()
    elif device == "cpu":
        result = torch.Tensor(arr)
    else:
        raise RuntimeError(f"Device type {device} unkown.")

    if synchronize and device == "cuda":
        torch.cuda.synchronize()

    time_to_tensor = perf_counter() - time_start
    return result, time_to_tensor


def to_numpy(t: torch.Tensor, device: str, synchronize=False) -> np.ndarray:
    """
    Converts a torch tensor on CPU/GPU to a numpy array
    """

    time_start = perf_counter()
    if device == "cuda":
        result = t.cpu().detach().numpy()
    elif device == "cpu":
        result = t.detach().numpy()
    else:
        raise RuntimeError(f"Device type {device} unkown.")

    if synchronize and device == "cuda":
        torch.cuda.synchronize()

    time_to_numpy = perf_counter() - time_start
    return result, time_to_numpy
