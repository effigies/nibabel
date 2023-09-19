"""Point-set structures

Imaging data are sampled at points in space, and these points
can be described by coordinates.
These structures are designed to enable operations on sets of
points, as opposed to the data sampled at those points.

Abstractly, a point set is any collection of points, but there are
two types that warrant special consideration in the neuroimaging
context: grids and meshes.

A *grid* is a collection of regularly-spaced points. The canonical
examples of grids are the indices of voxels and their affine
projection into a reference space.

A *mesh* is a collection of points and some structure that enables
adjacent points to be identified. A *triangular mesh* in particular
uses triplets of adjacent vertices to describe faces.
"""
from __future__ import annotations

import math
import typing as ty
from dataclasses import dataclass, replace

import numpy as np

from nibabel.casting import able_int_type
from nibabel.fileslice import strided_scalar
from nibabel.spatialimages import SpatialImage

if ty.TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import Self

    _DType = ty.TypeVar('_DType', bound=np.dtype[ty.Any])


class CoordinateArray(ty.Protocol):
    ndim: int
    shape: tuple[int, int]

    @ty.overload
    def __array__(self, dtype: None = ..., /) -> np.ndarray[ty.Any, np.dtype[ty.Any]]:
        ...  # pragma: no cover

    @ty.overload
    def __array__(self, dtype: _DType, /) -> np.ndarray[ty.Any, _DType]:
        ...  # pragma: no cover


class HasMeshAttrs(ty.Protocol):
    coordinates: CoordinateArray
    triangles: CoordinateArray


@dataclass
class Pointset:
    """A collection of points described by coordinates.

    Parameters
    ----------
    coords : array-like
      (*N*, *n*) array with *N* being points and columns their *n*-dimensional coordinates
    affine : :class:`numpy.ndarray`
      Affine transform to be applied to coordinates array
    homogeneous : :class:`bool`
      Indicate whether the provided coordinates are homogeneous,
      i.e., homogeneous 3D coordinates have the form ``(x, y, z, 1)``
    """

    coordinates: CoordinateArray
    affine: np.ndarray
    homogeneous: bool = False

    # Force use of __rmatmul__ with numpy arrays
    __array_priority__ = 99

    def __init__(
        self,
        coordinates: CoordinateArray,
        affine: np.ndarray | None = None,
        homogeneous: bool = False,
    ):
        self.coordinates = coordinates
        self.homogeneous = homogeneous

        if affine is None:
            self.affine = np.eye(self.dim + 1)
        else:
            self.affine = np.asanyarray(affine)

        if self.affine.shape != (self.dim + 1,) * 2:
            raise ValueError(f'Invalid affine for {self.dim}D coordinates:\n{self.affine}')
        if np.any(self.affine[-1, :-1] != 0) or self.affine[-1, -1] != 1:
            raise ValueError(f'Invalid affine matrix:\n{self.affine}')

    @property
    def n_coords(self) -> int:
        """Number of coordinates

        Subclasses should override with more efficient implementations.
        """
        return self.coordinates.shape[0]

    @property
    def dim(self) -> int:
        """The dimensionality of the space the coordinates are in"""
        return self.coordinates.shape[1] - self.homogeneous

    def __rmatmul__(self, affine: np.ndarray) -> Self:
        """Apply an affine transformation to the pointset

        This will return a new pointset with an updated affine matrix only.
        """
        return replace(self, affine=np.asanyarray(affine) @ self.affine)

    def _homogeneous_coords(self):
        if self.homogeneous:
            return np.asanyarray(self.coordinates)

        ones = strided_scalar(
            shape=(self.coordinates.shape[0], 1),
            scalar=np.array(1, dtype=self.coordinates.dtype),
        )
        return np.hstack((self.coordinates, ones))

    def get_coords(self, *, as_homogeneous: bool = False):
        """Retrieve the coordinates

        Parameters
        ----------
        as_homogeneous : :class:`bool`
            Return homogeneous coordinates if ``True``, or Cartesian
            coordiantes if ``False``.

        name : :class:`str`
            Select a particular coordinate system if more than one may exist.
            By default, `None` is equivalent to `"world"` and corresponds to
            an RAS+ coordinate system.
        """
        ident = np.allclose(self.affine, np.eye(self.affine.shape[0]))
        if self.homogeneous == as_homogeneous and ident:
            return np.asanyarray(self.coordinates)
        coords = self._homogeneous_coords()
        if not ident:
            coords = (self.affine @ coords.T).T
        if not as_homogeneous:
            coords = coords[:, :-1]
        return coords


class TriangularMesh(Pointset):
    triangles: CoordinateArray

    def __init__(
        self,
        coordinates: CoordinateArray,
        triangles: CoordinateArray,
        affine: np.ndarray | None = None,
        homogeneous: bool = False,
    ):
        super().__init__(coordinates, affine=affine, homogeneous=homogeneous)
        self.triangles = triangles

    @classmethod
    def from_tuple(
        cls,
        mesh: tuple[CoordinateArray, CoordinateArray],
        affine: np.ndarray | None = None,
        homogeneous: bool = False,
    ) -> Self:
        return cls(mesh[0], mesh[1], affine=affine, homogeneous=homogeneous)

    @classmethod
    def from_object(
        cls,
        mesh: HasMeshAttrs,
        affine: np.ndarray | None = None,
        homogeneous: bool = False,
    ) -> Self:
        return cls(mesh.coordinates, mesh.triangles, affine=affine, homogeneous=homogeneous)

    @property
    def n_triangles(self):
        """Number of faces

        Subclasses should override with more efficient implementations.
        """
        return self.triangles.shape[0]

    def get_triangles(self):
        """Mx3 array of indices into coordinate table"""
        return np.asanyarray(self.triangles)

    def get_mesh(self, *, as_homogeneous: bool = False):
        return self.get_coords(as_homogeneous=as_homogeneous), self.get_triangles()


class CoordinateFamilyMixin(Pointset):
    def __init__(self, *args, **kwargs):
        self._coords = {}
        super().__init__(*args, **kwargs)

    def get_names(self):
        """List of surface names that can be passed to :meth:`with_name`"""
        return list(self._coords)

    def with_name(self, name: str) -> Self:
        new = replace(self, coordinates=self._coords[name])
        new._coords = self._coords
        return new

    def add_coordinates(self, name, coordinates):
        self._coords[name] = coordinates


class Grid(Pointset):
    r"""A regularly-spaced collection of coordinates

    This class provides factory methods for generating Pointsets from
    :class:`~nibabel.spatialimages.SpatialImage`\s and generating masks
    from coordinate sets.
    """

    @classmethod
    def from_image(cls, spatialimage: SpatialImage) -> Self:
        return cls(coordinates=GridIndices(spatialimage.shape[:3]), affine=spatialimage.affine)

    @classmethod
    def from_mask(cls, mask: SpatialImage) -> Self:
        mask_arr = np.bool_(mask.dataobj)
        return cls(
            coordinates=np.c_[np.nonzero(mask_arr)].astype(able_int_type(mask.shape)),
            affine=mask.affine,
        )

    def to_mask(self, shape=None) -> SpatialImage:
        if shape is None:
            shape = tuple(np.max(self.coordinates, axis=0)[: self.dim] + 1)
        mask_arr = np.zeros(shape, dtype='bool')
        mask_arr[tuple(np.asanyarray(self.coordinates)[:, : self.dim].T)] = True
        return SpatialImage(mask_arr, self.affine)


class GridIndices:
    """Class for generating indices just-in-time"""

    __slots__ = ('gridshape', 'dtype', 'shape')
    ndim = 2

    def __init__(self, shape, dtype=None):
        self.gridshape = shape
        self.dtype = dtype or able_int_type(shape)
        self.shape = (math.prod(self.gridshape), len(self.gridshape))

    def __repr__(self):
        return f'<{self.__class__.__name__}{self.gridshape}>'

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self.dtype

        axes = [np.arange(s, dtype=dtype) for s in self.gridshape]
        return np.reshape(np.meshgrid(*axes, copy=False, indexing='ij'), (len(axes), -1)).T


class SliceIndices:
    """Class for generating indices using slice notation"""

    def __init__(self, shape, dtype=None, homogeneous=False):
        self.shape = shape
        self.dtype = dtype or able_int_type(shape)
        self.homogeneous = homogeneous

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        if any(idx is None for idx in index):
            raise IndexError('newaxis/None unsupported')

        left, right = [], []
        shape = list(self.shape)

        # Pre-ellipsis
        add = left.append
        getdim = lambda: shape.pop(0)

        ellipses = False
        for subidx in index:
            if subidx is Ellipsis:
                if ellipses:
                    raise IndexError
                add = lambda x: right.insert(0, x)
                getdim = shape.pop
                ellipses = True
                continue

            dim = getdim()

            # Standard slicing
            if isinstance(subidx, slice):
                start = min(subidx.start or 0, dim)
                if start < 0:
                    start += dim
                stop = min(subidx.stop, dim) if subidx.stop is not None else dim
                if stop < 0:
                    stop += dim
                add(np.mgrid[start : stop : subidx.step].astype(self.dtype).reshape(-1, 1))
                continue

            subidx = np.asanyarray(subidx, dtype=self.dtype)
            if subidx.ndim == 1:
                subidx = subidx.reshape(-1, 1)

            if subidx.dtype == np.dtype('bool'):
                # Mask array
                add(np.c_[np.nonzero(subidx)].astype(self.dtype))
            elif np.issubdtype(subidx.dtype, np.integer):
                # Fancy indexing
                add(subidx.astype(self.dtype))
            else:
                raise IndexError(f'Unsupported index type: {index.dtype}')

            # Pop extra dimensions if multiple indexed
            for _ in subidx.shape[1:]:
                getdim()

        # Handle ellipses and partial slicing
        if shape:
            left.extend(np.mgrid[:dim].astype(self.dtype).reshape(-1, 1) for dim in shape)

        indices = left + right
        cumlens = np.cumprod([1] + [arr.shape[0] for arr in indices])
        n = cumlens[-1]

        #  /------- Repeat 4x
        #  |
        #  |  /---- Repeat 2x, tile 2x
        #  |  |
        #  |  |  /- Tile 4x
        #  X0 Y0 Z0
        #  X0 Y0 Z1
        #  X0 Y1 Z0
        #  X0 Y1 Z1
        #  X1 Y0 Z0
        #  X1 Y0 Z1
        #  X1 Y1 Z0
        #  X1 Y1 Z1
        #
        #  For 2x2x2, cumlens=[1, 2, 4, 8], so tiles=[1, 2, 4], reps=[4, 2, 1]
        columns = [
            np.tile(np.repeat(arr, nreps, axis=0), (ntiles, 1))
            for arr, ntiles, nreps in zip(indices, cumlens, n // cumlens[1:])
        ]
        if self.homogeneous:
            columns.append(np.ones((n, 1), np.int32))
        return np.hstack(columns)
