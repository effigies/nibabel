import typing as ty

import numpy as np
import pytest

import nibabel as nb
from nibabel import AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage
from nibabel.spatialimages import Affine, SpatialImage

if ty.TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x: ty.Any) -> None:
        pass


@pytest.mark.mypy_testing
@pytest.mark.parametrize("image_type", [SpatialImage, AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image, MGHImage])
def test_functions_with_affines(image_type: ty.Type[SpatialImage[Affine]]) -> None:
    img_with_affine = image_type(np.empty((5, 5, 5), dtype=np.float32), np.eye(4))

    # A function that requires an affine will raise an error if the affine is None
    ras_img = nb.as_closest_canonical(img_with_affine)
    reveal_type(ras_img)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]

    # Functions that do not require affines should preserve the affine type
    reveal_type(nb.squeeze_image(img_with_affine))  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]

    # Concatenating images works if all have matching affines
    reveal_type(
        nb.concat_images([img_with_affine] * 2, check_affines=False)  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]
    )
    reveal_type(nb.concat_images([img_with_affine] * 2))  # R: nibabel.spatialimages.SpatialImage[numpy.ndarray[builtins.tuple[builtins.int, ...], numpy.dtype[numpy.floating[Any]]]]


@pytest.mark.mypy_testing
@pytest.mark.parametrize("image_type", [SpatialImage, AnalyzeImage, Spm99AnalyzeImage, Spm2AnalyzeImage, Nifti1Image, Nifti2Image])
def test_functions_without_affines(image_type: ty.Type[SpatialImage[None]]) -> None:
    img_without_affine = SpatialImage(np.empty((5, 5, 5)), None)

    # A function that requires an affine will raise an error if the affine is None
    with pytest.raises(Exception):
        nb.as_closest_canonical(img_without_affine)  # E: Value of type variable "SpatialImgT" of "as_closest_canonical" cannot be "SpatialImage[None]"  [type-var]

    # Functions that do not require affines should preserve the affine type
    reveal_type(nb.squeeze_image(img_without_affine))  # R: nibabel.spatialimages.SpatialImage[None]

    # Concatenating images works if all are missing affines
    reveal_type(nb.concat_images([img_without_affine] * 2, check_affines=False))  # R: nibabel.spatialimages.SpatialImage[None]
    reveal_type(nb.concat_images([img_without_affine] * 2))  # R: nibabel.spatialimages.SpatialImage[None]


@pytest.mark.mypy_testing
def test_concat_images_mixed_affines() -> None:
    img_with_affine = SpatialImage(np.empty((5, 5, 5)), np.eye(4))
    img_without_affine = SpatialImage(np.empty((5, 5, 5)), None)

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
