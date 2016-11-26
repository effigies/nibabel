# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from collections import namedtuple as nt

import numpy as np

from ..optpkg import optional_package
from ..viewers import OrthoSlicer3D

from numpy.testing.decorators import skipif
from numpy.testing import assert_array_equal, assert_equal

from nose.tools import assert_raises, assert_true, assert_false

# Need at least MPL 1.3 for viewer tests.
matplotlib, has_mpl, _ = optional_package('matplotlib', min_version='1.3')

needs_mpl = skipif(not has_mpl, 'These tests need matplotlib')
if has_mpl:
    matplotlib.use('Agg')


@needs_mpl
def test_viewer():
    # Test viewer
    plt = optional_package('matplotlib.pyplot')[0]
    a = np.sin(np.linspace(0, np.pi, 20))
    b = np.sin(np.linspace(0, np.pi*5, 30))
    data = (np.outer(a, b)[..., np.newaxis] * a)[:, :, :, np.newaxis]
    data = data * np.array([1., 2.])  # give it a # of volumes > 1
    v = OrthoSlicer3D(data)
    assert_array_equal(v.position, (0, 0, 0))
    assert_true('OrthoSlicer3D' in repr(v))

    # fake some events, inside and outside axes
    v._on_scroll(nt('event', 'button inaxes key')('up', None, None))
    for ax in (v._axes[0], v._axes[3]):
        v._on_scroll(nt('event', 'button inaxes key')('up', ax, None))
    v._on_scroll(nt('event', 'button inaxes key')('up', ax, 'shift'))
    # "click" outside axes, then once in each axis, then move without click
    v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, None, 1))
    for ax in v._axes:
        v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, ax, 1))
    v._on_mouse(nt('event', 'xdata ydata inaxes button')(0.5, 0.5, None, None))
    v.set_volume_idx(1)
    v.cmap = 'hot'
    v.clim = (0, 3)
    assert_raises(ValueError, OrthoSlicer3D.clim.fset, v, (0.,))  # bad limits
    assert_raises(ValueError, OrthoSlicer3D.cmap.fset, v, 'foo')  # wrong cmap

    # decrement/increment volume numbers via keypress
    v.set_volume_idx(1)  # should just pass
    v._on_keypress(nt('event', 'key')('-'))  # decrement
    assert_equal(v._data_idx[3], 0)
    v._on_keypress(nt('event', 'key')('+'))  # increment
    assert_equal(v._data_idx[3], 1)
    v._on_keypress(nt('event', 'key')('-'))
    v._on_keypress(nt('event', 'key')('='))  # alternative increment key
    assert_equal(v._data_idx[3], 1)

    v.close()
    v._draw()  # should be safe

    # Manually set value limits
    vlim = np.array([-20, 20])
    v = OrthoSlicer3D(data, vlim=vlim)
    assert_array_equal(v._clim, vlim)
    for im in v._ims:
        assert_array_equal(im.get_clim(), vlim)
    assert_array_equal(v._axes[3].get_ylim(), vlim)
    v.close()
    v1 = OrthoSlicer3D(data)
    v2 = OrthoSlicer3D(data, vlim=('1%', '99%'))
    assert_array_equal(v1.clim, v2.clim)
    v2.close()
    v2 = OrthoSlicer3D(data, vlim=('2%', '98%'))
    assert_false(np.array_equal(v1.clim, v2.clim))
    v2.close()
    v1.close()

    # non-multi-volume
    v = OrthoSlicer3D(data[:, :, :, 0])
    v._on_scroll(nt('event', 'button inaxes key')('up', v._axes[0], 'shift'))
    v._on_keypress(nt('event', 'key')('escape'))
    v.close()

    # complex input should raise a TypeError prior to figure creation
    assert_raises(TypeError, OrthoSlicer3D,
                  data[:, :, :, 0].astype(np.complex64))

    # other cases
    fig, axes = plt.subplots(1, 4)
    plt.close(fig)
    v1 = OrthoSlicer3D(data, axes=axes)
    aff = np.array([[0, 1, 0, 3], [-1, 0, 0, 2], [0, 0, 2, 1], [0, 0, 0, 1]],
                   float)
    v2 = OrthoSlicer3D(data, affine=aff, axes=axes[:3])
    # bad data (not 3+ dim)
    assert_raises(ValueError, OrthoSlicer3D, data[:, :, 0, 0])
    # bad affine (not 4x4)
    assert_raises(ValueError, OrthoSlicer3D, data, affine=np.eye(3))
    assert_raises(TypeError, v2.link_to, 1)
    v2.link_to(v1)
    v2.link_to(v1)  # shouldn't do anything
    v1.close()
    v2.close()

    # Test various slicers
    OrthoSlicer3D(data, slicer=Ellipsis).close()
    # Slice time dimension
    OrthoSlicer3D(data, slicer=(Ellipsis, slice(None, None, 2))).close()
    OrthoSlicer3D(data, slicer=(Ellipsis, 0)).close()
    OrthoSlicer3D(data, slicer=(Ellipsis, [0])).close()
    # Slice spatial dimensions
    OrthoSlicer3D(data, slicer=(slice(0, 1), slice(0, 1), slice(0, 1))).close()
    OrthoSlicer3D(data, slicer=(slice(0, -1), slice(0, -1), slice(0, -1))).close()
    # Fail if we slice too thin
    assert_raises(ValueError, OrthoSlicer3D, data, slicer=(Ellipsis, 0, 0))


@needs_mpl
def test_viewer_vlims():
    shape = (5, 6, 7)
    data = np.linspace(-1, 2, np.prod(shape)).reshape(shape)
    p1, p3, p94, p99 = np.percentile(data, [1, 3, 94, 99])
    assert_array_equal(OrthoSlicer3D(data).clim, [p1, p99])
    for vlim, expected in ((None, (p1, p99)),
                           ((0, 1), (0, 1)),
                           ((0, '94%'), (0, p94)),
                           (('3%', 1), (p3, 1)),
                           (('3%', '94%'), (p3, p94)),
                          ):
        assert_array_equal(OrthoSlicer3D(data, vlim=vlim).clim, expected)
        # Default
        ov = OrthoSlicer3D(data)
        assert_array_equal(ov.clim, [p1, p99])
        # Changed
        ov.clim = vlim
        assert_array_equal(ov.clim, expected)
