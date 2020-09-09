# -*- coding: utf-8 -*-
"""Test the supply curve SupplyCurveAggregation with friction surface.

Created on Wed Jun 19 15:37:05 2019

@author: gbuster
"""
import h5py
import numpy as np
import pytest
import os
import warnings

from reV.supply_curve.points import SupplyCurveExtent
from reV.supply_curve.exclusions import ExclusionMaskFromDict, FrictionMask
from reV.supply_curve.sc_aggregation import SupplyCurveAggregation
from reV import TESTDATADIR

EXCL_FPATH = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
FRICTION_FPATH = os.path.join(TESTDATADIR, 'ri_exclusions/ri_friction.h5')
FRICTION_DSET = 'ri_friction_surface'
GEN = os.path.join(TESTDATADIR, 'gen_out/ri_my_pv_gen.h5')
AGG_BASELINE = os.path.join(TESTDATADIR, 'sc_out/baseline_agg_summary.csv')
TM_DSET = 'techmap_nsrdb'
RES_CLASS_DSET = 'ghi_mean-means'
RES_CLASS_BINS = [0, 4, 100]
DATA_LAYERS = {'pct_slope': {'dset': 'ri_srtm_slope',
                             'method': 'mean'},
               'reeds_region': {'dset': 'ri_reeds_regions',
                                'method': 'mode'},
               'padus': {'dset': 'ri_padus',
                         'method': 'mode'}}

EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': True},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': True}}

RESOLUTION = 64
EXTENT = SupplyCurveExtent(EXCL_FPATH, resolution=RESOLUTION)
EXCL = ExclusionMaskFromDict(EXCL_FPATH, EXCL_DICT)
FRICTION = FrictionMask(FRICTION_FPATH, FRICTION_DSET)


def test_friction_mask():
    """Test the friction mask on known quantities."""

    x = FRICTION[slice(700, 800), slice(300, 400)].mean()
    assert x == 20.0, 'Friction for region should be 20.0, but is {}'.format(x)

    x = FRICTION[slice(300, 400), slice(100, 300)].mean()
    assert x == 1.0, 'Friction for nodata should be 1.0, but is {}'.format(x)

    x = FRICTION[slice(300, 400), slice(800, 900)].mean()
    assert x == 1.0, 'Friction for nodata should be 1.0, but is {}'.format(x)

    x = FRICTION[slice(0, 10), slice(0, 10)].mean()
    assert x == 10.0, 'Friction for region should be 10.0, but is {}'.format(x)

    x = FRICTION[slice(354, 360), slice(456, 460)].mean()
    diff = (x - 1.2275390625) / x
    m = 'Friction for region should be 1.228, but is {}'.format(x)
    assert diff < 0.00001, m


@pytest.mark.parametrize('gid', [100, 114, 130, 181])
def test_agg_friction(gid):
    """Test SC Aggregation with friction by checking friction factors and LCOE
    against a hand calc."""

    warnings.filterwarnings('ignore')

    for gid in [100, 114, 130, 181]:
        s = SupplyCurveAggregation.summary(EXCL_FPATH, GEN, TM_DSET,
                                           excl_dict=EXCL_DICT,
                                           res_class_dset=RES_CLASS_DSET,
                                           res_class_bins=RES_CLASS_BINS,
                                           data_layers=DATA_LAYERS,
                                           resolution=RESOLUTION,
                                           gids=[gid], max_workers=1,
                                           friction_fpath=FRICTION_FPATH,
                                           friction_dset=FRICTION_DSET)

        row_slice, col_slice = EXTENT.get_excl_slices(gid)

        test_e = EXCL[row_slice, col_slice]
        test_f = FRICTION[row_slice, col_slice]
        x = test_e * test_f
        x = x.flatten()
        x = x[(x != 0)]
        mean_friction = x.mean()

        m = ('SC point gid {} does not match mean friction hand calc'
             .format(gid))
        assert s['mean_friction'].values[0] == mean_friction, m
        m = ('SC point gid {} does not match mean LCOE with friction hand calc'
             .format(gid))
        assert np.allclose(s['mean_lcoe_friction'],
                           s['mean_lcoe'] * mean_friction), m


def make_friction_file():
    """Script to make a test friction file"""
    import matplotlib.pyplot as plt
    import shutil
    shutil.copy(EXCL, FRICTION_FPATH)
    with h5py.File(FRICTION_FPATH, 'a') as f:
        f[FRICTION_DSET] = f['ri_srtm_slope']
        attrs = dict(f[FRICTION_DSET].attrs)
        print(attrs)
        shape = f[FRICTION_DSET].shape
        data = np.random.lognormal(mean=0.2, sigma=0.2, size=shape)
        data = data.astype(np.float32)
        plt.hist(data.flatten())
        plt.show()
        plt.close()

        data[:, 0:100, 0:100] = 10.0
        data[:, 700:800, 300:400] = 20.0
        data[:, 300:400, 800:900] = -9999
        data[:, 300:400, 100:300] = -9999

        f[FRICTION_DSET][...] = data
        for d in f:
            print(d, f[d].shape, f[d].dtype)
            if d not in [FRICTION_DSET, 'latitude', 'longitude']:
                del f[d]

    with h5py.File(FRICTION_FPATH, 'r') as f:
        for d in f:
            print(d, f[d].shape, f[d].dtype)
        out = f[FRICTION_DSET][...]

    assert np.allclose(data, out)


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':
    execute_pytest()
