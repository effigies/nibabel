""" Finite geometric structures

Neuroimaging applications make use of finite geometric constructs
"""

class Pointset:
    """ A point set is a set of coordinates in 3D space. """

    def get_coords(self, name=None):
        """ Return coordinate data in RAS+ space

        Parameters
        ----------
        name : str
            A family of related coordinates may be retrieved by name.

        Returns
        -------
        coords : (N, 3) array-like
            Coordinates in RAS+ space
        """
        raise NotImplementedError

    @property
    def n_coords(self):
        """ Number of coordinates (vertices)

        The default implementation loads coordinates. Subclasses may
        override with more efficient implementations.
        """
        return self.get_coords().shape[0]


class GeometryCollection:
    def __init__(self, structures=()):
        self._structures = dict(structures)

    def get_structure(self, name):
        return self._structures[name]

    @property
    def names(self):
        return list(self._structures)

    @classmethod
    def from_spec(klass, pathlike):
        """ Load a collection of geometries from a specification, broadly construed. """
        raise NotImplementedError


class GeometrySequence(GeometryCollection, Pointset):
    def __init__(self, structures=()):
        super().__init__(structures)
        self._indices = {}
        next_index = 0
        for name, struct in self._structures.items():
            end = next_index + struct.n_coords
            self._indices[name] = slice(next_index, end)
            next_index = end + 1

    def get_indices(self, *names):
        if len(names) == 1:
            return self._indices[name]
        return [self._indices[name] for name in names]

    # def get_structures(self, *, names=None, indices=None):
    #     """ We probably want some way to get a subset of structures """

    def get_coords(self, name=None):
        return np.vstack([struct.get_coords(name=name)
                          for struct in self._structures.values()])

    @property
    def n_coords(self):
        return sum(struct.n_coords for struct in self._structures.values())
