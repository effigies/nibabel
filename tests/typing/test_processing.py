import typing as ty

import numpy as np
import pytest

from nibabel import AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage
from nibabel.spatialimages import Affine, SpatialImage
from nibabel import processing as nbp

if ty.TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x: ty.Any) -> None:
        pass


@pytest.mark.mypy_testing
@pytest.mark.parametrize("image_type", [SpatialImage, AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage])
def test_resample_from_to(image_type: ty.Type[SpatialImage[Affine]]) -> None:
    img = image_type(np.empty((5, 5, 5), dtype=np.float32), np.eye(4))

    reveal_type(img)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]

    resampled_img1 = nbp.resample_from_to(img, ((6, 6, 6), np.eye(4)))
    reveal_type(resampled_img1)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img1, Nifti1Image)

    resampled_img2 = nbp.resample_from_to(img, ((6, 6, 6), np.eye(4)), out_class=None)
    reveal_type(resampled_img2)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img2, image_type)

    resampled_img3 = nbp.resample_from_to(img, ((6, 6, 6), np.eye(4)), out_class=Nifti1Image)
    reveal_type(resampled_img3)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img3, Nifti1Image)

    resampled_img4 = nbp.resample_from_to(img, ((6, 6, 6), np.eye(4)), out_class=AnalyzeImage)

    # Unfortunately, there's no way to annotate it so `AnalyzeImage` produces `AnalyzeImage[Affine]`,
    # at least, without listing them out by hand.
    # The common cases are Nifti1Image and None, and those get the affine right.
    reveal_type(resampled_img4)  # R: nibabel.analyze.AnalyzeImage[Any]
    reveal_type(resampled_img4.affine)  # R: Any
    assert isinstance(resampled_img4, AnalyzeImage)
    assert isinstance(resampled_img4.affine, np.ndarray)

    # Check that positional and kwargs are handled
    # Explicitly passing AnalyzeImage[Affine] works
    resampled_img5 = nbp.resample_from_to(img, ((6, 6, 6), np.eye(4)), 3, mode="constant", out_class=AnalyzeImage[Affine])
    reveal_type(resampled_img5)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img5, AnalyzeImage)


@pytest.mark.mypy_testing
@pytest.mark.parametrize("image_type", [SpatialImage, AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage])
def test_resample_to_output(image_type: ty.Type[SpatialImage[Affine]]) -> None:
    img = image_type(np.empty((5, 5, 5), dtype=np.float32), np.eye(4))

    reveal_type(img)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]

    resampled_img1 = nbp.resample_to_output(img, 6)
    reveal_type(resampled_img1)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img1, Nifti1Image)

    resampled_img2 = nbp.resample_to_output(img, (6, 6, 6), out_class=None)
    reveal_type(resampled_img2)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img2, image_type)

    resampled_img3 = nbp.resample_to_output(img, 6, out_class=Nifti1Image)
    reveal_type(resampled_img3)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img3, Nifti1Image)

    resampled_img4 = nbp.resample_to_output(img, 6, out_class=AnalyzeImage)

    # See first test for discussion about Any
    reveal_type(resampled_img4)  # R: nibabel.analyze.AnalyzeImage[Any]
    reveal_type(resampled_img4.affine)  # R: Any
    assert isinstance(resampled_img4, AnalyzeImage)
    assert isinstance(resampled_img4.affine, np.ndarray)

    # Check that positional and kwargs are handled
    resampled_img5 = nbp.resample_to_output(img, 6, 3, mode="constant", out_class=AnalyzeImage[Affine])
    reveal_type(resampled_img5)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img5, AnalyzeImage)


@pytest.mark.mypy_testing
@pytest.mark.parametrize("image_type", [SpatialImage, AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage])
def test_smooth_image(image_type: ty.Type[SpatialImage[Affine]]) -> None:
    img = image_type(np.empty((5, 5, 5), dtype=np.float32), np.eye(4))

    reveal_type(img)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]

    resampled_img1 = nbp.smooth_image(img, 6)
    reveal_type(resampled_img1)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img1, Nifti1Image)

    resampled_img2 = nbp.smooth_image(img, (6, 6, 6), out_class=None)
    reveal_type(resampled_img2)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img2, image_type)

    resampled_img3 = nbp.smooth_image(img, 6, out_class=Nifti1Image)
    reveal_type(resampled_img3)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img3, Nifti1Image)

    resampled_img4 = nbp.smooth_image(img, 6, out_class=AnalyzeImage)

    # See first test for discussion about Any
    reveal_type(resampled_img4)  # R: nibabel.analyze.AnalyzeImage[Any]
    reveal_type(resampled_img4.affine)  # R: Any
    assert isinstance(resampled_img4, AnalyzeImage)
    assert isinstance(resampled_img4.affine, np.ndarray)

    # Check that positional and kwargs are handled
    resampled_img5 = nbp.smooth_image(img, 6, "constant", cval=0.0, out_class=AnalyzeImage[Affine])
    reveal_type(resampled_img5)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img5, AnalyzeImage)


@pytest.mark.mypy_testing
@pytest.mark.parametrize("image_type", [SpatialImage, AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage])
def test_conform(image_type: ty.Type[SpatialImage[Affine]]) -> None:
    img = image_type(np.empty((5, 5, 5), dtype=np.float32), np.eye(4))

    reveal_type(img)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]

    resampled_img1 = nbp.conform(img)
    reveal_type(resampled_img1)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img1, image_type)

    resampled_img2 = nbp.conform(img, (6, 6, 6), out_class=Nifti1Image[Affine])
    reveal_type(resampled_img2)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    assert isinstance(resampled_img2, Nifti1Image)

    resampled_img3 = nbp.conform(img, (6, 6, 6), order=3, out_class=Nifti1Image)
    reveal_type(resampled_img3)  # R: nibabel.nifti1.Nifti1Image[Any]
    assert isinstance(resampled_img3, Nifti1Image)
