"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(local_rank=0):
    """
    Setup the device for training on MPS (Metal Performance Shaders).
    """
    if th.backends.mps.is_available():
        device = th.device("mps")
        print("Using MPS backend.")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        raise RuntimeError("MPS (Metal) backend is not available on this system.")

    return device


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file
    """

    with bf.BlobFile(path, "rb") as f:
        data = f.read()

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
