'''
Unit tests for excel_reader
'''

from atlas_building_tools.exceptions import AtlasBuildingToolsWarning
from pathlib import Path
import warnings
import re
import numpy as np
import numpy.testing as npt

import atlas_building_tools.densities.excel_reader as tested


DATA_PATH = Path(
    Path(__file__).parent.parent.parent, 'atlas_building_tools', 'app', 'data'
)


def test_compute_kim_et_al_neuron_densities():
    with warnings.catch_warnings(record=True) as warnings_:
        dataframe = tested.compute_kim_et_al_neuron_densities(
            Path(DATA_PATH, 'mmc3.xlsx')
        )
        warnings_ = [
            w for w in warnings_ if isinstance(w.message, AtlasBuildingToolsWarning)
        ]
        regions_with_nd = []
        regions_with_invalid_full_name = []
        nd_warning_regexp = re.compile(r'Region (.*) has "N/D" values')
        full_name_regexp = re.compile(r'Region (.*) has no valid full name')
        for warning in warnings_:
            found = nd_warning_regexp.search(str(warning))
            if found is not None:
                regions_with_nd.append(found.group(1))
            found = full_name_regexp.search(str(warning))
            if found is not None:
                regions_with_invalid_full_name.append(found.group(1))
        npt.assert_array_equal(
            regions_with_nd,
            [
                'SF',
                'TRS',
                'PVH',
                'PVHm',
                'PVHmm',
                'PVHpm',
                'PVHpml',
                'PVHpmm',
                'PVHp',
                'PVHap',
                'PVHmpd',
                'PVHpv',
                'PVa',
                'PVi',
                'KF',
            ],
        )
        npt.assert_array_equal(regions_with_invalid_full_name, ['IB'])
        expected_columns = [
            'Full name',
            'PV',
            'PV_stddev',
            'SST',
            'SST_stddev',
            'VIP',
            'VIP_stddev',
        ]
        npt.assert_array_equal(dataframe.columns, expected_columns)
        assert len(dataframe.index) == 824
        assert (
            len({'grey', 'CH', 'CTX', 'IB', 'SF', 'TRS'} - set(dataframe.index))
            == 0
        )
        assert (
            len(
                {'Whole brain', 'Cerebrum', 'Cerebral cortex', 'Olfactory areas'}
                - set(dataframe['Full name'])
            )
            == 0
        )
        assert not np.any(np.isnan(dataframe['SST']))
        assert not np.any(np.isnan(dataframe['SST_stddev']))
        assert not np.any(np.isnan(dataframe['VIP']))
        assert not np.any(np.isnan(dataframe['VIP_stddev']))
        assert np.count_nonzero(np.isnan(dataframe['PV'])) == 15
        assert np.count_nonzero(np.isnan(dataframe['PV_stddev'])) == 15

        for column in expected_columns[1:]:
            mask = ~np.isnan(dataframe[column])
            assert np.all(dataframe[column].to_numpy()[mask] >= 0.0)

        assert np.allclose(
            dataframe['PV'][0], (6000.52758114279 + 5831.1654080927) / 2.0
        )
        assert np.allclose(
            dataframe['SST'][1], (5083.93779315662 + 5422.81140933669) / 2.0
        )
        assert np.allclose(
            dataframe['VIP'][2], (1935.39495313035 + 2013.81692802911) / 2.0
        )
