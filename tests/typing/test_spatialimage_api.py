import typing as ty

import numpy as np
import pytest

import nibabel as nb
from nibabel.analyze import AnalyzeImage
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage

if ty.TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x: ty.Any) -> None:
        pass


@pytest.mark.mypy_testing
def test_SpatialImageAffines() -> None:
    """Test type hints for the SpatialImage class."""
    img_with_affine = SpatialImage(np.empty((5, 5, 5)), np.eye(4))
    img_without_affine = SpatialImage(np.empty((5, 5, 5)), None)

    reveal_type(img_with_affine)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(img_without_affine)  # R: nibabel.spatialimages.SpatialImage[None]

    # A function that requires an affine will raise an error if the affine is None
    ras_img = nb.as_closest_canonical(img_with_affine)
    reveal_type(ras_img)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    with pytest.raises(Exception):
        nb.as_closest_canonical(img_without_affine)  # E: Value of type variable "SpatialImgT" of "as_closest_canonical" cannot be "SpatialImage[None]"  [type-var]

    # Functions that do not require affines should preserve the affine type
    reveal_type(nb.squeeze_image(img_with_affine))  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(nb.squeeze_image(img_without_affine))  # R: nibabel.spatialimages.SpatialImage[None]

    reveal_type(
        nb.concat_images([img_with_affine, img_with_affine], check_affines=False)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    )
    reveal_type(nb.concat_images([img_without_affine] * 2, check_affines=False))  # R: nibabel.spatialimages.SpatialImage[None]
    # In practice, the first affine is used, but that's difficult to annotate, so we lose specificity
    reveal_type(
        nb.concat_images(  # R: nibabel.spatialimages.SpatialImage[Union[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]], None]]
            [img_with_affine, img_without_affine], check_affines=False
        )
    )

    # If check_affines=True (default), then type checking and execution can fail unless all check_affines
    # either exist or are absent
    reveal_type(nb.concat_images([img_with_affine] * 2))  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(nb.concat_images([img_without_affine] * 2))  # R: nibabel.spatialimages.SpatialImage[None]
    with pytest.raises(ValueError):
        nb.concat_images(  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "SpatialImage[ndarray[tuple[int, ...], dtype[float64]] | None]"
            [img_with_affine, img_without_affine]
        )
    with pytest.raises(ValueError):
        nb.concat_images(  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "SpatialImage[ndarray[tuple[int, ...], dtype[float64]] | None]"
            [img_without_affine, img_with_affine]
        )

    mixed_list = [img_with_affine, img_without_affine]
    with pytest.raises(ValueError):
        nb.concat_images(mixed_list)  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "SpatialImage[ndarray[tuple[int, ...], dtype[float64]] | None]"  [type-var]


@pytest.mark.mypy_testing
def test_AnalyzeImageAffines() -> None:
    img_with_affine = AnalyzeImage(np.empty((5, 5, 5)), np.eye(4))
    img_without_affine = AnalyzeImage(np.empty((5, 5, 5)), None)

    reveal_type(img_with_affine)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(img_without_affine)  # R: nibabel.analyze.AnalyzeImage[None]

    reveal_type(img_with_affine)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(img_without_affine)  # R: nibabel.analyze.AnalyzeImage[None]

    # A function that requires an affine will raise an error if the affine is None
    ras_img = nb.as_closest_canonical(img_with_affine)
    reveal_type(ras_img)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    with pytest.raises(Exception):
        nb.as_closest_canonical(img_without_affine)  # E: Value of type variable "SpatialImgT" of "as_closest_canonical" cannot be "AnalyzeImage[None]"  [type-var]

    # Functions that do not require affines should preserve the affine type
    reveal_type(nb.squeeze_image(img_with_affine))  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(nb.squeeze_image(img_without_affine))  # R: nibabel.analyze.AnalyzeImage[None]

    reveal_type(
        nb.concat_images([img_with_affine, img_with_affine], check_affines=False)  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    )
    reveal_type(nb.concat_images([img_without_affine] * 2, check_affines=False))  # R: nibabel.analyze.AnalyzeImage[None]
    # In practice, the first affine is used, but that's difficult to annotate, so we lose specificity
    reveal_type(
        nb.concat_images(  # R: nibabel.analyze.AnalyzeImage[Union[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]], None]]
            [img_with_affine, img_without_affine], check_affines=False
        )
    )

    # If check_affines=True (default), then type checking and execution can fail unless all check_affines
    # either exist or are absent
    reveal_type(nb.concat_images([img_with_affine] * 2))  # R: nibabel.analyze.AnalyzeImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(nb.concat_images([img_without_affine] * 2))  # R: nibabel.analyze.AnalyzeImage[None]
    with pytest.raises(ValueError):
        nb.concat_images(  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "AnalyzeImage[ndarray[tuple[int, ...], dtype[float64]] | None]"
            [img_with_affine, img_without_affine]
        )
    with pytest.raises(ValueError):
        nb.concat_images(  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "AnalyzeImage[ndarray[tuple[int, ...], dtype[float64]] | None]"
            [img_without_affine, img_with_affine]
        )

    mixed_list = [img_with_affine, img_without_affine]
    with pytest.raises(ValueError):
        nb.concat_images(mixed_list)  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "AnalyzeImage[ndarray[tuple[int, ...], dtype[float64]] | None]"


@pytest.mark.mypy_testing
def test_Nifti1ImageAffines() -> None:
    img_with_affine = Nifti1Image(np.empty((5, 5, 5)), np.eye(4))
    img_without_affine = Nifti1Image(np.empty((5, 5, 5)), None)

    reveal_type(img_with_affine)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(img_without_affine)  # R: nibabel.nifti1.Nifti1Image[None]

    reveal_type(img_with_affine)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(img_without_affine)  # R: nibabel.nifti1.Nifti1Image[None]

    # A function that requires an affine will raise an error if the affine is None
    ras_img = nb.as_closest_canonical(img_with_affine)
    reveal_type(ras_img)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    with pytest.raises(Exception):
        nb.as_closest_canonical(img_without_affine)  # E: Value of type variable "SpatialImgT" of "as_closest_canonical" cannot be "Nifti1Image[None]"  [type-var]

    # Functions that do not require affines should preserve the affine type
    reveal_type(nb.squeeze_image(img_with_affine))  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(nb.squeeze_image(img_without_affine))  # R: nibabel.nifti1.Nifti1Image[None]

    reveal_type(
        nb.concat_images([img_with_affine, img_with_affine], check_affines=False)  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    )
    reveal_type(nb.concat_images([img_without_affine] * 2, check_affines=False))  # R: nibabel.nifti1.Nifti1Image[None]
    # In practice, the first affine is used, but that's difficult to annotate, so we lose specificity
    reveal_type(
        nb.concat_images(  # R: nibabel.nifti1.Nifti1Image[Union[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]], None]]
            [img_with_affine, img_without_affine], check_affines=False
        )
    )

    # If check_affines=True (default), then type checking and execution can fail unless all check_affines
    # either exist or are absent
    reveal_type(nb.concat_images([img_with_affine] * 2))  # R: nibabel.nifti1.Nifti1Image[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.float64]]]
    reveal_type(nb.concat_images([img_without_affine] * 2))  # R: nibabel.nifti1.Nifti1Image[None]
    with pytest.raises(ValueError):
        nb.concat_images(  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "Nifti1Image[ndarray[tuple[int, ...], dtype[float64]] | None]"
            [img_with_affine, img_without_affine]
        )
    with pytest.raises(ValueError):
        nb.concat_images(  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "Nifti1Image[ndarray[tuple[int, ...], dtype[float64]] | None]"
            [img_without_affine, img_with_affine]
        )

    mixed_list = [img_with_affine, img_without_affine]
    with pytest.raises(ValueError):
        nb.concat_images(mixed_list)  # E: Value of type variable "_OneSpatialImgT" of "concat_images" cannot be "Nifti1Image[ndarray[tuple[int, ...], dtype[float64]] | None]"
