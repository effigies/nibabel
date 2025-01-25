import typing as ty

import numpy as np
import pytest

from nibabel.nifti1 import Nifti1Image

if ty.TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x: ty.Any) -> None:
        pass


@pytest.mark.mypy_testing
def test_Nifti1ImageType() -> None:
    img = Nifti1Image(np.empty((5, 5, 5)), np.eye(4))

    reveal_type(img)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(img.header)  # R: nibabel.nifti1.Nifti1Header
    reveal_type(img.affine)  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]

    reveal_type(img.dataobj)  # R: nibabel.arrayproxy.ArrayLike
    reveal_type(img.get_fdata())  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype=np.float32))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype=np.float64))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype=np.dtype(np.float32)))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype=np.dtype(np.float64)))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype=np.dtype("f4")))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype=np.dtype("f8")))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]
    reveal_type(img.get_fdata(dtype="f4"))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]
    reveal_type(img.get_fdata(dtype="f8"))  # R: numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]

    reveal_type(img.shape)  # R: builtins.tuple[builtins.int, ...]
    reveal_type(img.ndim)  # R: builtins.int
