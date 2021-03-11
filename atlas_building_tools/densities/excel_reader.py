'''Utility functions to retrieve cell counts and cell densities from the scientific literature
when stored in excel spread sheets.

Lexicon: AIBS stands for Allen Institute for Brain Science
    https://alleninstitute.org/what-we-do/brain-science/
'''

from warnings import warn
from typing import TYPE_CHECKING, Union
import numpy as np
import pandas as pd

from atlas_building_tools.exceptions import AtlasBuildingToolsWarning

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path


def compute_kim_et_al_neuron_densities(
    inhibitory_neuron_densities_path: Union[str, 'Path']
) -> 'pd.DataFrame':
    """
    Extract from excel file and average over gender the densities of the cells reacting to
    PV, SST and VIP in every AIBS region of the mouse brain.

    The following markers are used by Kim et al., in 'Brain-wide Maps Reveal Stereotyped Cell-'
    'Type-Based Cortical Architecture and Subcortical Sexual Dimorphism' to detect inhibitory
    neurons:
        * parvalbumin (PV)
        * somatostatin (SST)
        * vasoactive intestinal peptide (VIP).
    The markers PV, SST and VIP are only expressed in neurons.

    Note: A handful of region full names are not compliant with the AIBS glossary and should be
    addressed by the consumer if need be.
    There are 15 regions (e.g., SF or TRS, see unit tests) which have values "N/D" in the PV male
    column. These values are handled as NaNs. One region, namely IB, has an invalid full name,
    that is, the float nan.

    Args:
        inhibitory_neuron_densities_path: path to the excel document mm3c.xls of the supplementary
            materials of 'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture
            and Subcortical Sexual Dimorphism' by Kim et al., 2017.
            https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc3.xlsx

    Returns: pandas.DataFrame of the form (values are fake)
                                   Full name     PV   PV_stddev  SST  SST_stddev   VIP  VIP_stddev
        ROI
        grey                     Whole brain     289  0.1        235  0.1         451  0.1
        CH                          Cerebrum     660  0.1        138  0.1         425  0.1
        CTX                  Cerebral cortex     627  0.2        106  0.3         424  0.1
        CTXpl                 Cortical plate     619  0.3        999  0.4         414  0.2
        Isocortex                  Isocortex     533  0.4        594  0.1         280  0.1
        ...                            ...         ...              ...              ...
        The columns of PV, SST and VIP contain the densities of the corresponding immunoreactive
        cells for each AIBS acronym listed in the DataFrame index.
        Each column has been obtained by averaging the original male and female columns of the input
        worksheet. For instance PV_stddev is row-wise the mean value of the standard deviations
        PV_male_stddev and PV_female_stddev.

    """
    # Retrieve the PV, SST and VIP densities for both male and female specimen.
    kim_et_al_mmc3 = pd.read_excel(
        str(inhibitory_neuron_densities_path),
        sheet_name='Sheet1',
        header=None,
        names=[
            'ROI',
            'Full name',
            'PV_male',
            'PV_male_stddev',
            'PV_female',
            'PV_female_stddev',
            'SST_male',
            'SST_male_stddev',
            'SST_female',
            'SST_female_stddev',
            'VIP_male',
            'VIP_male_stddev',
            'VIP_female',
            'VIP_female_stddev',
        ],
        usecols='A,B,' + 'D,E,F,G,' + 'H,I,J,K,' + 'L,M,N,O',
        skiprows=[0, 1],
        engine='openpyxl',
    ).set_index('ROI')
    for region_name in kim_et_al_mmc3.index:
        full_name = kim_et_al_mmc3.loc[region_name][0]
        if not isinstance(full_name, str) or not full_name:
            warn(
                f'Region {region_name} has no valid full name. '
                f'Found: {full_name} of type {type(full_name)}',
                AtlasBuildingToolsWarning,
            )
        nd_mask = kim_et_al_mmc3.loc[region_name] == 'N/D'
        if np.any(nd_mask):
            warn(
                f'Region {region_name} has "N/D" values in the following columns: '
                f'{list(kim_et_al_mmc3.columns[nd_mask])}. The "N/D" values will be handled as'
                ' NaNs.',
                AtlasBuildingToolsWarning,
            )

    kim_et_al_mmc3.replace('N/D', 'NaN', inplace=True)

    # Average over the 2 genders
    data_frame = pd.DataFrame(
        {'Full name': kim_et_al_mmc3['Full name']}, index=kim_et_al_mmc3.index
    )
    for marker in ['PV', 'SST', 'VIP']:
        densities = np.array([
            np.asarray(kim_et_al_mmc3[f'{marker}_{gender}'], dtype=float)
            for gender in ['male', 'female']
        ])
        data_frame[marker] = np.sum(densities, axis=0) / 2.0  # 2 genders
        standard_deviations = np.array([
            np.asarray(kim_et_al_mmc3[f'{marker}_{gender}_stddev'], dtype=float)
            for gender in ['male', 'female']
        ])
        data_frame[f'{marker}_stddev'] = np.sum(standard_deviations, axis=0) / 2.0  # 2 genders

    return data_frame
