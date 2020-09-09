# -*- coding: utf-8 -*-
"""
QA/QC tests
"""
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from reV import TESTDATADIR
from reV.qa_qc.summary import SummarizeH5, SummarizeSupplyCurve

H5_FILE = os.path.join(TESTDATADIR, 'gen_out', 'ri_wind_gen_profiles_2010.h5')
SC_TABLE = os.path.join(TESTDATADIR, 'sc_out', 'sc_full_out_1.csv')
SUMMARY_DIR = os.path.join(TESTDATADIR, 'qa_qc')


@pytest.mark.parametrize('dataset', ['cf_mean', 'cf_profile', None])
def test_summarize(dataset):
    """Run QA/QC Summarize and compare with baseline"""

    summary = SummarizeH5(H5_FILE)

    if dataset is None:
        baseline = os.path.join(SUMMARY_DIR,
                                'ri_wind_gen_profiles_2010_summary.csv')
        baseline = pd.read_csv(baseline)
        test = summary.summarize_means()
    elif dataset == 'cf_mean':
        baseline = os.path.join(SUMMARY_DIR, 'cf_mean_summary.csv')
        baseline = pd.read_csv(baseline, index_col=0)
        test = summary.summarize_dset(
            dataset, process_size=None, max_workers=1)
    elif dataset == 'cf_profile':
        baseline = os.path.join(SUMMARY_DIR, 'cf_profile_summary.csv')
        baseline = pd.read_csv(baseline, index_col=0)
        test = summary.summarize_dset(
            dataset, process_size=None, max_workers=1)

    assert_frame_equal(test, baseline, check_dtype=False)


def test_sc_summarize():
    """Run QA/QC Summarize and compare with baseline"""
    test = SummarizeSupplyCurve(SC_TABLE).supply_curve_summary()
    baseline = os.path.join(SUMMARY_DIR,
                            'sc_full_out_1_summary.csv')
    baseline = pd.read_csv(baseline, index_col=0)

    assert_frame_equal(test, baseline, check_dtype=False)
