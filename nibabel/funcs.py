# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Processor functions for images"""

from __future__ import annotations

import typing as ty

import numpy as np

from .loadsave import load_as
from .orientations import OrientationError, io_orientation
from .spatialimages import SpatialImage

if ty.TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal as L

    import numpy.typing as npt

    from .filename_parser import FileSpec
    from .spatialimages import Affine, AnySpatialImgT, SpatialImgT

    _OneSpatialImgT = ty.TypeVar(
        '_OneSpatialImgT', bound=SpatialImage[Affine] | SpatialImage[None]
    )


def squeeze_image(img: AnySpatialImgT) -> AnySpatialImgT:
    """Return image, remove axes length 1 at end of image shape

    For example, an image may have shape (10,20,30,1,1).  In this case
    squeeze will result in an image with shape (10,20,30).  See doctests
    for further description of behavior.

    Parameters
    ----------
    img : ``SpatialImage``

    Returns
    -------
    squeezed_img : ``SpatialImage``
       Copy of img, such that data, and data shape have been squeezed,
       for dimensions > 3rd, and at the end of the shape list

    Examples
    --------
    >>> import nibabel as nf
    >>> shape = (10,20,30,1,1)
    >>> data = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    >>> affine = np.eye(4)
    >>> img = nf.Nifti1Image(data, affine)
    >>> img.shape == (10, 20, 30, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 20, 30)
    True

    If the data are 3D then last dimensions of 1 are ignored

    >>> shape = (10,1,1)
    >>> data = np.arange(np.prod(shape), dtype='int32').reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (10, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (10, 1, 1)
    True

    Only *final* dimensions of 1 are squeezed

    >>> shape = (1, 1, 5, 1, 2, 1, 1)
    >>> data = data.reshape(shape)
    >>> img = nf.ni1.Nifti1Image(data, affine)
    >>> img.shape == (1, 1, 5, 1, 2, 1, 1)
    True
    >>> img2 = squeeze_image(img)
    >>> img2.shape == (1, 1, 5, 1, 2)
    True
    """
    klass = img.__class__
    shape = img.shape
    slen = len(shape)
    if slen < 4:
        return klass.from_image(img)
    for bdim in shape[3::][::-1]:
        if bdim == 1:
            slen -= 1
        else:
            break
    if slen == len(shape):
        return klass.from_image(img)
    shape = shape[:slen]
    data = np.asanyarray(img.dataobj).reshape(shape)
    return klass(data, img.affine, img.header, img.extra)


# concat_images is one of those functions that might do too many things.
# It will load images from filenames, and we cannot determine the specific
# image in that case.
# It can also check that affines are equal, which work if the affines are all
# the same or all absent.
# The following overloads attempt to ensure that the type of the images
# (e.g., Nifti1Image) is preserved, and the presence of the affine is preserved,
# when possible.


# If check_affines is True and all images are of the same type (including
# affine presence), type is preserved
@ty.overload
def concat_images(
    images: Sequence[_OneSpatialImgT],
    check_affines: bool = True,
    axis: int | None = None,
) -> _OneSpatialImgT: ...
# If check_affines is False and all images are of the same type (permitting
# mixed affines), the type is preserved and presence of affine is indeterminate
@ty.overload
def concat_images(
    images: Sequence[AnySpatialImgT],
    check_affines: L[False],
    axis: int | None = None,
) -> AnySpatialImgT: ...
# If images are loaded from file, affines will be initialized, but type
# can't be known. Sorry.
@ty.overload
def concat_images(
    images: Sequence[FileSpec],
    check_affines: bool = True,
    axis: int | None = None,
) -> SpatialImage[Affine]: ...
# If filenames and images are passed in a mixed list, we can say the output
# will be a SpatialImage.
@ty.overload
def concat_images(
    images: Sequence[FileSpec | SpatialImage[ty.Any]],
    check_affines: L[False],
    axis: int | None = None,
) -> SpatialImage[ty.Any]: ...


# There is a missing option here, if the type system would permit it:
# Images with affines and filenames could be concatenated and we would know
# there would be an affine, regardless of whether it's checked.
# I couldn't find a way to add it to the overloads at the bottom of the list,
# and inserting it above reduced the specificity of the earlier overloads.
# It seems like a pretty rare case, in any event.


def concat_images(
    images: Sequence[SpatialImage[ty.Any] | FileSpec],
    check_affines: bool = True,
    axis: int | None = None,
) -> SpatialImage[ty.Any]:
    r"""Concatenate images in list to single image, along specified dimension

    Parameters
    ----------
    images : sequence
       sequence of ``SpatialImage`` or filenames of the same dimensionality\s
    check_affines : {True, False}, optional
       If True, then check that all the affines for `images` are nearly
       the same, raising a ``ValueError`` otherwise.  Default is True
    axis : None or int, optional
        If None, concatenates on a new dimension.  This requires all images to
        be the same shape.  If not None, concatenates on the specified
        dimension.  This requires all images to be the same shape, except on
        the specified dimension.

    Returns
    -------
    concat_img : ``SpatialImage``
       New image resulting from concatenating `images` across last
       dimension
    """
    loaded: list[SpatialImage[Affine | None]] = [
        img if isinstance(img, SpatialImage) else load_as(img, SpatialImage) for img in images
    ]
    n_imgs = len(loaded)
    if n_imgs == 0:
        raise ValueError('Cannot concatenate an empty list of images.')
    img0 = loaded[0]
    affine = img0.affine
    header = img0.header
    klass = img0.__class__
    shape0 = img0.shape
    n_dim = len(shape0)
    if axis is None:
        # collect images in output array for efficiency
        out_shape = (n_imgs,) + shape0
        out_data = np.empty(out_shape)
    else:
        # collect images in list for use with np.concatenate
        out_data = [None] * n_imgs  # type: ignore[assignment]
    # Get part of shape we need to check inside loop
    idx_mask = np.ones((n_dim,), dtype=bool)
    if axis is not None:
        idx_mask[axis] = False
    masked_shape = np.array(shape0)[idx_mask]
    for i, img in enumerate(loaded):
        if len(img.shape) != n_dim:
            raise ValueError(f'Image {i} has {len(img.shape)} dimensions, image 0 has {n_dim}')
        if not np.all(np.array(img.shape)[idx_mask] == masked_shape):
            raise ValueError(
                f'shape {img.shape} for image {i} not compatible with '
                f'first image shape {shape0} with axis == {axis}'
            )
        if check_affines and not np.all(img.affine == affine):
            raise ValueError(f'Affine for image {i} does not match affine for first image')
        # Do not fill cache in image if it is empty
        out_data[i] = np.asanyarray(img.dataobj)

    if axis is None:
        out_data = np.rollaxis(out_data, 0, out_data.ndim)
    else:
        out_data = np.concatenate(out_data, axis=axis)

    return klass(out_data, affine, header)


def four_to_three(img: AnySpatialImgT) -> list[AnySpatialImgT]:
    """Create 3D images from 4D image by slicing over last axis

    Parameters
    ----------
    img :  image
       4D image instance of some class with methods ``get_data``,
       ``header`` and ``affine``, and a class constructor
       allowing klass(data, affine, header)

    Returns
    -------
    imgs : list
       list of 3D images
    """
    arr = np.asanyarray(img.dataobj)
    header = img.header
    affine = img.affine
    image_maker = img.__class__
    if arr.ndim != 4:
        raise ValueError('Expecting four dimensions')
    imgs = []
    for i in range(arr.shape[3]):
        arr3d = arr[..., i]
        img3d = image_maker(arr3d, affine, header)
        imgs.append(img3d)
    return imgs


def as_closest_canonical(img: SpatialImgT, enforce_diag: bool = False) -> SpatialImgT:
    """Return `img` with data reordered to be closest to canonical

    Canonical order is the ordering of the output axes.

    Parameters
    ----------
    img : ``spatialimage``
    enforce_diag : {False, True}, optional
       If True, before transforming image, check if the resulting image
       affine will be close to diagonal, and if not, raise an error

    Returns
    -------
    canonical_img : ``spatialimage``
       Version of `img` where the underlying array may have been
       reordered and / or flipped so that axes 0,1,2 are those axes in
       the input data that are, respectively, closest to the output axis
       orientation.  We modify the affine accordingly.  If `img` is
       already has the correct data ordering, we just return `img`
       unmodified.
    """
    # Get the image class to transform the data for us
    img = img.as_reoriented(io_orientation(img.affine))

    # however, the affine may not be diagonal
    if enforce_diag and not _aff_is_diag(img.affine):
        raise OrientationError('Transformed affine is not diagonal')

    return img


def _aff_is_diag(aff: npt.NDArray[np.floating]) -> bool:
    """Utility function returning True if affine is nearly diagonal"""
    rzs_aff = aff[:3, :3]
    return np.allclose(rzs_aff, np.diag(np.diag(rzs_aff)))
