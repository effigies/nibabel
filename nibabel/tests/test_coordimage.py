import os
from pathlib import Path

import nibabel as nb
from nibabel import coordimage as ci
from nibabel import pointset as ps
from nibabel.tests.nibabel_data import get_nibabel_data

from .test_pointset import FreeSurferHemisphere

CIFTI2_DATA = Path(get_nibabel_data()) / 'nitest-cifti2'


class FreeSurferSubject(ci.GeometryCollection):
    @classmethod
    def from_subject(klass, subject_id, subjects_dir=None):
        """Load a FreeSurfer subject by ID"""
        if subjects_dir is None:
            subjects_dir = os.environ['SUBJECTS_DIR']
        return klass.from_spec(Path(subjects_dir) / subject_id)

    @classmethod
    def from_spec(klass, pathlike):
        """Load a FreeSurfer subject from its directory structure"""
        subject_dir = Path(pathlike)
        surfs = subject_dir / 'surf'
        structures = {
            'lh': FreeSurferHemisphere.from_filename(surfs / 'lh.white'),
            'rh': FreeSurferHemisphere.from_filename(surfs / 'rh.white'),
        }
        subject = klass(structures)
        subject._subject_dir = subject_dir
        return subject


class CaretSpec(ci.GeometryCollection):
    @classmethod
    def from_spec(klass, pathlike):
        from nibabel.cifti2.caretspec import CaretSpecFile

        csf = CaretSpecFile.from_filename(pathlike)
        structures = {
            df.structure: df.uri
            for df in csf.data_files
            if df.selected  # Use selected to avoid overloading for now
        }
        wbspec = klass(structures)
        wbspec._specfile = csf
        return wbspec


def test_Cifti2Image_as_CoordImage():
    ones = nb.load(CIFTI2_DATA / 'ones.dscalar.nii')
    assert ones.shape == (1, 91282)
    cimg = ci.CoordinateImage.from_image(ones)
    assert cimg.shape == (91282, 1)

    caxis = cimg.coordaxis
    assert len(caxis) == 91282
    assert caxis[...] is caxis
    assert caxis[:] is caxis

    subaxis = caxis[:100]
    assert len(subaxis) == 100
    assert len(subaxis.parcels) == 1
    subaxis = caxis[100:]
    assert len(subaxis) == len(caxis) - 100
    assert len(subaxis.parcels) == len(caxis.parcels)
    subaxis = caxis[100:-100]
    assert len(subaxis) == len(caxis) - 200
    assert len(subaxis.parcels) == len(caxis.parcels)

    caxis.get_indices('CIFTI_STRUCTURE_CORTEX_LEFT')
