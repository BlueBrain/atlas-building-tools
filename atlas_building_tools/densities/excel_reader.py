"""Utility functions to retrieve measurements pertaining to cell density and stored in excel files

Measurements related to cell density are read and put under a uniform format so as to ease-off
computations made by their consumers of the densities module.

The two excel files to be handled are:
* mm3c.xls from the supplementary materials of
    'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical Architecture and Subcortical
    Sexual Dimorphism' by Kim et al., 2017.
    https://ars.els-cdn.com/content/image/1-s2.0-S0092867417310693-mmc3.xlsx

* atlas_building_tools/app/data/gaba_papers.xlsx, a compilation of measurements from the scientifc
    literature made by Rodarie Dimitri (BBP).

This module extract measurements from the above two files and collect them into a unique data frame
with the following columns:

* brain region (str), a mouse brain region name, not necessarily compliant with AIBS 1.json file.
    Thus some filtering must be done when working with AIBS annotated files.
* cell type (str, e.g, 'PV+' for cells reacting to parvalbumin, 'SST+' for cells reacting to
    somatostatin, 'inhibitory neuron' for non-specfic inhibitory neurons)
* measurement (float)
* standard deviation (non-negative float)
* measurement type (str), see MEASUREMENT_TYPES below
* measurement unit (str), see MEASUREMENT_UNITS below
* comment (str), a comment on how the measurement has been obtained
* source title (str), the title of the article where the measurement can be exracted
* specimen age (str, e.g., '8 week old', 'P56', '3 month old'), age of the mice used to obtain the
    measurement

Note: the measurements of gaba_papers.xlsx which are not expressed directly in terms of
number of cells per mm^3 (cell density) have been extracted manually and saved into
atlas_building_tools/app/data/non_density_measurements.csv under the above format.


Lexicon: AIBS stands for Allen Institute for Brain Science
    https://alleninstitute.org/what-we-do/brain-science/
"""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
import numpy.testing as npt
import pandas as pd

from atlas_building_tools.exceptions import AtlasBuildingToolsWarning

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path

    from pandas import DataFrame
    from voxcell import RegionMap

L = logging.getLogger(__name__)


MEASUREMENT_TYPES = {
    # For a given brain region R and a given cell type T:
    0: "cell density",  # number of cells of type T per mm^3 in R
    1: "neuron proportion",  # number of cells of type T / number of neurons in R
    2: "cell proportion",  # number of cells of type T / number of cells in R
    3: "cell count per slice",  # number of cells of type T per slice of R, see MEASUREMENT_UNITS
}

MEASUREMENT_UNITS = {
    "cell density": "number of cells per mm^3",
    "neuron proportion": "None",
    "cell proportion": "None",
    "cell count per slice": "number of cells per 5 micrometer-thick slice",
}


def compute_kim_et_al_neuron_densities(
    inhibitory_neuron_densities_path: Union[str, "Path"]
) -> "pd.DataFrame":
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

    Note: A handful of region 'Full name's are not compliant with the AIBS glossary and are to be
    addressed by the consumer.
    There are 15 regions (e.g., SF or TRS, see unit tests) which have values "N/D" in the PV male
    column. These values are handled as NaNs. One region, namely IB, has an invalid full name,
    that is, the float nan, when loaded with pd.read_excel.

    Args:
        inhibitory_neuron_densities_path: path to the excel document mm3c.xls of the
            supplementary materials of 'Brain-wide Maps Reveal Stereotyped Cell-Type-Based Cortical
             Architecture and Subcortical Sexual Dimorphism' by Kim et al., 2017.
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
        sheet_name="Sheet1",
        header=None,
        names=[
            "ROI",
            "Full name",
            "PV_male",
            "PV_male_stddev",
            "PV_female",
            "PV_female_stddev",
            "SST_male",
            "SST_male_stddev",
            "SST_female",
            "SST_female_stddev",
            "VIP_male",
            "VIP_male_stddev",
            "VIP_female",
            "VIP_female_stddev",
        ],
        usecols="A,B," + "D:G," + "H:K," + "L:O",
        skiprows=[0, 1],
        engine="openpyxl",
    ).set_index("ROI")
    for acronym in kim_et_al_mmc3.index:
        full_name = kim_et_al_mmc3.loc[acronym][0]
        if not isinstance(full_name, str) or not full_name:
            warn(
                f"Region with acronym {acronym} has no valid full name. "
                f"Found: {full_name} of type {type(full_name)}",
                AtlasBuildingToolsWarning,
            )
        nd_mask = kim_et_al_mmc3.loc[acronym] == "N/D"
        if np.any(nd_mask):
            warn(
                "[Reading Kim et al. 2017, mmc3.xlsx] "
                f'Region {full_name} with acronym {acronym} has "N/D" values in the following '
                f'columns: {list(kim_et_al_mmc3.columns[nd_mask])}. The "N/D" values will be '
                "handled as NaNs.",
                AtlasBuildingToolsWarning,
            )

    kim_et_al_mmc3.replace("N/D", "NaN", inplace=True)

    # Average over the 2 genders
    data_frame = pd.DataFrame(
        {"Full name": kim_et_al_mmc3["Full name"]}, index=kim_et_al_mmc3.index
    )
    for marker in ["PV", "SST", "VIP"]:
        densities = np.array(
            [
                np.asarray(kim_et_al_mmc3[f"{marker}_{gender}"], dtype=float)
                for gender in ["male", "female"]
            ]
        )
        data_frame[marker] = np.sum(densities, axis=0) / 2.0  # 2 genders
        standard_deviations = np.array(
            [
                np.asarray(kim_et_al_mmc3[f"{marker}_{gender}_stddev"], dtype=float)
                for gender in ["male", "female"]
            ]
        )
        data_frame[f"{marker}_stddev"] = np.sum(standard_deviations, axis=0) / 2.0  # 2 genders

    return data_frame


def _get_aibs_region_names(region_map: "RegionMap") -> Set[str]:
    """
    Retrieve the names of every region in `region_map`.

    Args:
        region_map: RegionMap object to navigate the brain regions hierarchy
            instantiated with the 1.json hierarchy file from AIBS.

    Returns:
        Set of strings containing the names of all regions represented in
        `region_map`.

    """
    aibs_region_ids = region_map.find(
        "Basic cell groups and regions", attr="name", with_descendants=True
    )

    return {region_map.get(id_, "name") for id_ in aibs_region_ids}


def _set_metadata_columns(
    dataframe: "pd.DataFrame",
    bibliography: "pd.DataFrame",
    comments: "pd.DataFrame",
) -> None:
    """
    Add a source title a comment to every measurement in `dataframe`

    The input `dataframe` is modified in-place.

    In `dataframe`, the rows corresponding to the same article have no valid 'source title',
    'comment' nor 'specimen age' except the first one. This function makes sure every row is
    set with approriate values in the aforementioned columns.

    Args:
        dataframe: DataFrame obtained when reading the worksheets 'GAD67 densities' or 'PV-SST-VIP'
            of gaba_papers.xlsx.
        bibliography: DataFrame of the form
                source title                              specimen age
            6   'A Cell Atlas for the Mouse Brain'        11 week old
            12  'Structural and Functional Aberrations'   56 days
            ...  ...                                      ...
        comments: DataFrame of the form
                comment
            6   'Divide Table 1 row by pi'
            17  'Add Table 1 to Table 2'
            ...  ...
    """

    def _fill_in_the_gaps(
        dataframe: "pd.DataFrame", collapsed_blocks: "pd.DataFrame", header: str
    ) -> None:
        """
        Assign to each line block represented in `collapsed_blocks` the value of the first line
        of the block wrt the column `header`.

        The input `dataframe` is modified in-line.

        A line block is for instance defined by a source title. It corresponds to merged cells in
        an excel worksheet. Each line of a block is a measurement that was found in the same article
        for the same specimen. However, when reading the excel worksheet with pandas, only the
        first line of the block has the article name in its 'source title' column and the specimen
        age in its 'specimen age' column because of merged cells. The subsequent lines of the block
        are filled with nan. This function replaces those nans by the 'source title' value of the
        first line in the block.

        The same applies to the 'comment' column. Source title blocks and comment blocks do not
        always coincide.
        """
        columns: Dict[str, List[str]] = defaultdict(list)
        for (start, end) in zip(collapsed_blocks.index, collapsed_blocks.index[1:]):
            columns[header].extend([dataframe[header][start]] * (end - start))
        # Handle the tail of the column
        delta = len(dataframe.index) - len(columns[header])
        dataframe[header] = (
            columns[header] + [dataframe[header][collapsed_blocks.index[-1]]] * delta
        )

    # Source titles and specimen ages define the same blocks of merged cells
    for header in ["source title", "specimen age"]:
        _fill_in_the_gaps(dataframe, bibliography, header)

    _fill_in_the_gaps(dataframe, comments, "comment")


def _set_measurement_type_and_unit_columns(
    dataframe: "pd.DataFrame", code_filter: Optional[List[int]] = None
) -> None:
    """
    Set the values of the 'measurement type' and 'measurement unit' columns.

    The assigment is based on a integer code in the range [0, 3], see MEASUREMENT_TYPES and
    MEASUREMENT_UNITS.

    Args:
        dataframe: DataFrame obtained when reading the worksheets 'GAD67 densities' or 'PV-SST-VIP'
            of gaba_papers.xlsx.
        code_filter: list of integers in the range [0, 3] used to filter-in measurements.
            By default, only the measurements with code 0, i.e., cell density measurments
            (number of cells per mm^3) are selected.

    """
    if code_filter is None:
        code_filter = [0]  # 'cell density' only
    mask = dataframe["measurement type"].isin(code_filter)
    dataframe.drop(dataframe.index[~mask], inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dataframe["measurement type"] = [
        MEASUREMENT_TYPES[code] for code in dataframe["measurement type"]
    ]
    dataframe["measurement unit"] = [
        MEASUREMENT_UNITS[type_] for type_ in dataframe["measurement type"]
    ]


def _create_bibliography(dataframe: "pd.DataFrame") -> "pd.DataFrame":
    """
    Create a helper bibliograph dataframe.

    Args:
        dataframe: DataFrame obtained when reading the worksheets 'GAD67 densities' or 'PV-SST-VIP'
            of gaba_papers.xlsx.

    Returns:
        a DataFrame of the form
                source title                             comment                      specimen age
            6   'A Cell Atlas for the Mouse Brain'       'Divide Table 1 row by pi'   11 weeks
            12  'Structural and Functional Aberrations'  'Add Table 1 to table 2'     56 days
            ...  ...                                      ...                         ...
    """
    return pd.DataFrame({"source title": dataframe["source title"]}).dropna()


def _create_collapsed_comments(dataframe: "pd.DataFrame") -> "pd.DataFrame":
    """
    Create a helper comments dataframe.

    Args:
        dataframe: DataFrame obtained when reading the worksheets 'GAD67 densities' or 'PV-SST-VIP'
            of gaba_papers.xlsx.

    Returns:
        a DataFrame of the form
                comment
            6   'Divide Table 1 row by pi'
            17  'Add Table 1 to Table 2'
            ...  ...
    """
    return pd.DataFrame({"comment": dataframe["comment"]}).dropna()


def _enforce_column_types(dataframe: "pd.Dataframe", has_source: bool = True) -> None:
    """
    Prescribe the data types of the `dataframe` columns.

    The `dataframe` is modified in-place.

    Args:
        dataframe: DataFrame obtained when reading the worksheets 'GAD67 densities' and 'PV-SST-VIP'
            of gaba_papers.xlsx, or 'Sheet 1' from mmc3.xlsx.
        has_source: if True, the types of the columns of 'source title', 'comment' and
            'specimen age' are also prescribed.
    """
    dataframe.astype(
        {
            "brain region": str,
            "cell type": str,
            "measurement": float,
            "measurement unit": str,
            "standard deviation": float,
            "measurement type": str,
        },
        copy=False,
    )

    if has_source:
        dataframe.astype(
            {
                "comment": str,
                "source title": str,
                "specimen age": str,
            },
            copy=False,
        )


def read_inhibitory_neuron_measurement_compilation(
    measurements_path: Union[str, "Path"]
) -> Tuple["pd.DataFrame", Set[str]]:
    """
    Read the neuron densities of the worksheet 'GAD67 densities' in gaba_papers.xlsx

    Args:
        region_map: RegionMap object to navigate the brain regions hierarchy.
            Assumed to be instantiated with AIBS 1.json.
        measurements_path: path to the file gaba_papers.xlsx

    Returns:
        a pd.DataFrame with the columns listed in the description at the top of
        this module.
    """
    # Reading takes several seconds, possibly due to formulaes interpretation
    L.info("Loading excel worksheet ...")
    inhibitory_neurons_worksheet = pd.read_excel(
        str(measurements_path),
        sheet_name="GAD67 densities",
        header=None,
        names=[
            "brain region",
            "measurement",
            "standard deviation",
            "comment",
            "source title",
            "specimen age",
            "measurement type",
        ],
        usecols="A:G",
        skiprows=[0],
        engine="openpyxl",
    )

    L.info("Operating on dataframe ...")
    _set_measurement_type_and_unit_columns(inhibitory_neurons_worksheet)
    bibliography = _create_bibliography(inhibitory_neurons_worksheet)
    comments = _create_collapsed_comments(inhibitory_neurons_worksheet)
    _set_metadata_columns(inhibitory_neurons_worksheet, bibliography, comments)
    # Inhibitory neurons are assumed to coincide with the neurons which are
    # expressing GAD67, i.e., the GAD67+ cells.
    inhibitory_neurons_worksheet["cell type"] = "inhibitory neuron"
    inhibitory_neurons_worksheet.reset_index(drop=True, inplace=True)
    _enforce_column_types(inhibitory_neurons_worksheet)

    return inhibitory_neurons_worksheet


def _stack_pv_sst_vip_measurements(dataframe: "pd.DataFrame") -> "pd.DataFrame":
    """
    Turns measurements stored into columns with labels PV, SST or VIP into the uniform
    flat format described at the top of the module.

    NA measurements are dropped.

    Args:
        dataframe: DataFrame obtained when reading the worksheets 'PV-SST-VIP'
            of gaba_papers.xlsx and 'Sheet 1' from mmc3.xlsx.

    Returns:
        DataFrame with the columns listed at the top of this module.
    """
    result = pd.DataFrame()
    marker_columns = {"PV", "SST", "VIP", "PV_stddev", "SST_stddev", "VIP_stddev"}
    for marker in ["PV", "SST", "VIP"]:
        to_drop = list(marker_columns - {marker, marker + "_stddev"})
        marker_dataframe = dataframe.drop(columns=to_drop)
        marker_dataframe.rename(
            columns={marker: "measurement", marker + "_stddev": "standard deviation"},
            inplace=True,
        )
        marker_dataframe["cell type"] = marker + "+"
        result = result.append(marker_dataframe)

    result.reset_index(drop=True, inplace=True)
    na_measurement_mask = result["measurement"].isna()
    result.drop(result.index[na_measurement_mask], inplace=True)
    result.reset_index(drop=True, inplace=True)

    return result


def read_pv_sst_vip_measurement_compilation(
    measurements_path: Union[str, "Path"]
) -> "pd.DataFrame":
    """
    Read the neuron densities of the worksheet 'PV-SST-VIP' in gaba_papers.xlsx

    Args:
        region_map: RegionMap object to navigate the brain regions hierarchy.
            Assumed to be instantiated with AIBS 1.json.
        measurements_path: path to the file gaba_papers.xlsx

    Returns:
        dataframe is a pd.DataFrame with the columns listed in the description at the top of
        this module.
    """
    pv_sst_vip_neurons_worksheet = pd.read_excel(
        str(measurements_path),
        sheet_name="PV-SST-VIP",
        header=None,
        names=[
            "brain region",
            "PV",
            "PV_stddev",
            "SST",
            "SST_stddev",
            "VIP",
            "VIP_stddev",
            "comment",
            "source title",
            "specimen age",
            "measurement type",
        ],
        usecols="A:K",
        skiprows=[0],
        engine="openpyxl",
    )

    _set_measurement_type_and_unit_columns(pv_sst_vip_neurons_worksheet)
    bibliography = _create_bibliography(pv_sst_vip_neurons_worksheet)
    comments = _create_collapsed_comments(pv_sst_vip_neurons_worksheet)
    _set_metadata_columns(pv_sst_vip_neurons_worksheet, bibliography, comments)
    pv_sst_vip_neurons_worksheet = _stack_pv_sst_vip_measurements(pv_sst_vip_neurons_worksheet)
    _enforce_column_types(pv_sst_vip_neurons_worksheet)

    return pv_sst_vip_neurons_worksheet


def read_homogenous_neuron_type_regions(measurements_path: Union[str, "Path"]) -> "pd.DataFrame":
    """
    Read the region list of the worksheet 'Full inhibexc regions' in gaba_papers.xlsx

    Args:
        measurements_path: path to the file gaba_papers.xlsx

    Returns:
        pd.DataFrame with two columns: 'brain region' and 'cell type'. A cell type value
        is either 'ihibitory' or 'excitatory' and applies to every cell in the region.
    """

    homogenous_regions_worksheet = pd.read_excel(
        str(measurements_path),
        sheet_name="Fully inhibexc regions",
        names=["brain region", "comment"],
        header=None,
        usecols="A,D",
        skiprows=[0],
        engine="openpyxl",
    )
    excitatory_mask = homogenous_regions_worksheet["comment"] == "Purely excitatory region"
    homogenous_regions_worksheet["cell type"] = "inhibitory"
    homogenous_regions_worksheet["cell type"][excitatory_mask] = "excitatory"
    homogenous_regions_worksheet.drop(columns=["comment"], inplace=True)

    return homogenous_regions_worksheet


def _enforce_aibs_nomenclature(
    region_map: "RegionMap", dataframe: "pd.DataFrame"
) -> "pd.DataFrame":
    """
    Make region names appearing in mmc3.xlsx AIBS compliant whenever possible.

    Args:
        region_map: RegionMap object to navigate the brain regions hierarchy.
            Assumed to be instantiated with AIBS 1.json.
        dataframe: the DataFrame obtained when reading 'Sheet 1' of mmc3.xlsx.

    Returns:
        a dataframe where brain region names have been made valid wrt AIBS 1.json file each
        time it is possible. The remaining unhandled invalid names are not removed.
    """

    def _replace_accents(region_name: str) -> str:
        return region_name.replace("È", "e")

    dataframe.loc[
        dataframe["Full name"] == "Whole brain", "Full name"
    ] = "Basic cell groups and regions"

    # Missing layer prefix
    acav6 = "Anterior cingulate area, ventral part, 6"
    dataframe.loc[dataframe["Full name"] == acav6, "Full name"] = acav6 + "a"
    copied_row = dataframe[dataframe["Full name"] == acav6 + "a"].copy()
    copied_row["Full name"] = acav6 + "b"
    dataframe = dataframe.append(copied_row)

    # Fixing accents
    dataframe["Full name"] = [_replace_accents(name) for name in dataframe["Full name"]]

    # Most of the issues comme from names ending with 'layer 6' instead of ending with 'layer 6a'
    # or 'layer 6b' as in AIBS 1.json.
    # When such a problematic name is encountered, we remove it and insert its 'layer 6a' and/or
    # 'layer 6b' counterpart , if they exist.
    aibs_region_names = _get_aibs_region_names(region_map)
    invalid_region_names = set(dataframe["Full name"]) - aibs_region_names
    layer_6_names = set(
        name for name in aibs_region_names if ("layer 6" in name) or ("Layer 6" in name)
    )
    # "Dorsal peduncular area, layer 6a" is the only AIBS name listed in the "brain region"
    # column as "Dorsal peduncular area, layer 6" but with no "Dorsal peduncular area, layer 6b"
    # counterpart in AIBS 1.json.

    # Insert valid AIBS counterparts
    handled_invalid_names = set()
    for region_name in layer_6_names:
        for invalid_name in invalid_region_names:
            if invalid_name in region_name:
                handled_invalid_names.add(invalid_name)
                indices = dataframe.index[dataframe["Full name"] == invalid_name]
                assert (
                    len(indices) == 1
                ), f"Found more than one index for {invalid_name}, expected only one."
                row = dataframe.loc[indices[0]]
                new_row = pd.DataFrame({column: [row[column]] for column in dataframe.columns})
                new_row["Full name"] = region_name
                dataframe = dataframe.append(new_row)

    # Remove the invalid region names which have been fixed
    indices = dataframe.index[dataframe["Full name"].isin(list(handled_invalid_names))]
    dataframe.drop(indices, inplace=True)

    return dataframe


def read_kim_et_al_neuron_densities(
    region_map: "RegionMap", inhibitory_neuron_densities_path: Union[str, "Path"]
) -> "pd.DataFrame":
    """
    Read the neuron densities of the worksheet 'Sheet 1' in mmc3.xlsx

    Args:
        region_map: RegionMap object to navigate the AIBS brain regions hierarchy.
            Used to handle invalid AIBS region names.
        inhibitory_neuron_densities_path: path to the file mmc3.xlsx

    Returns:
        a pd.DataFrame with the columns listed in the description at the top of this module.
    """

    densities_dataframe = compute_kim_et_al_neuron_densities(inhibitory_neuron_densities_path)

    # Only one full name is missing (NaN): the entry corresponding to the ROI "IB", i.e.,
    # AIBS "Interbrain" region
    nan_fullname_mask = densities_dataframe["Full name"].isna()
    npt.assert_array_equal(densities_dataframe.index[nan_fullname_mask], ["IB"])
    densities_dataframe.loc[nan_fullname_mask, "Full name"] = "Interbrain"

    densities_dataframe.reset_index(drop=True, inplace=True)
    densities_dataframe = _enforce_aibs_nomenclature(region_map, densities_dataframe)
    densities_dataframe.rename(columns={"Full name": "brain region"}, inplace=True)
    densities_dataframe["measurement type"] = "cell density"
    densities_dataframe["measurement unit"] = "number of cells per mm^3"
    densities_dataframe = _stack_pv_sst_vip_measurements(densities_dataframe)
    _enforce_column_types(densities_dataframe, has_source=False)

    densities_dataframe["source title"] = (
        "Brain-wide Maps Reveal Stereotyped Cell-Type-Based"
        " Cortical Architecture and Subcortical Sexual Dimorphism"
    )
    densities_dataframe["comment"] = "Average over genders"
    densities_dataframe["specimen age"] = "8- to 10-week old"

    return densities_dataframe


def read_measurements(
    region_map: "RegionMap",
    mmc3_path: Union[str, "Path"],
    gaba_papers_path: Union[str, "Path"],
    non_density_measurements_path: Union[str, "Path"],
) -> "pd.DataFrame":
    """
    Read all cell density related measurements from file and returns a unique DataFrame
    containing them.

    The format of the output DataFrame is described at the top of this module.

    Args:
        region_map: RegionMap object to navigate the AIBS brain regions hierarchy
        mmc3_path: path to mmc3.xlsx
        gaba_papers_path: path to gaba_papers.xlsx
        non_density_measurements_path: path to the csv file containing the measurements of
            gaba_papers.xslx which are not expressed in number of cells per mm^3
            (e.g., cell proportions, number of cells per slice). These measurements have been
            extracted manually.

    Returns:
        a pd.DataFrame containing the measurements of mmc3.xlsx and gaba_papers.xlsx under the
        format described at the top of this module.

    """
    dataframe = read_kim_et_al_neuron_densities(region_map, mmc3_path)
    dataframe = dataframe.append(read_inhibitory_neuron_measurement_compilation(gaba_papers_path))
    dataframe = dataframe.append(read_pv_sst_vip_measurement_compilation(gaba_papers_path))
    dataframe = dataframe.append(pd.read_csv(non_density_measurements_path))
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe
