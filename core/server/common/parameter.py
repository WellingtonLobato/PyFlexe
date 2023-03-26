import numpy as np

from io import BytesIO
from typing import cast, List


def ndarrays_to_parameters(ndarrays: np.ndarray) -> List[bytes]:
    """Convert NumPy ndarrays to parameters object."""
    return [ndarray_to_bytes(ndarray) for ndarray in ndarrays]

def parameters_to_ndarrays(parameters: List[bytes]) -> np.ndarray:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters]

def bytes_to_ndarray(tensor: bytes) -> np.ndarray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)

def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()
