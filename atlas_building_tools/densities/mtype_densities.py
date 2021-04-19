"""
Create a density field for each mtype listed in app/data/mapping.tsv

Volumetric density nrrd files are created for each mtype listed in mapping.tsv.
This module re-use the overall excitatory and inhibitory neuron densities computed in
app/cell_densities.

Each mtype is assigned a neuron density profile, that is, a list of cells counts corresponding to
the layer slices (a. k. a. bins) defined in app/data/meta/layers.tsv.
The delination of the layer slices, or sub-layers, within the annotated 3D volume of the AIBS mouse
isocortex is based on the placement hints computed in app/placement_hints. Placement hints's
distance information allows us to split each layer into slices of approximately the same thickness
along the cortical axis, as presribed by app/data/meta/layers.tsv.

The input neuron density profiles have been obtained in
"A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex", 2019
by D. Keller et al.

Lexicon:
    * mtype: morphological type, e.g., L23_DBC, L5_TPC:C or L6_UPC (iscortex mtypes are listed in
        mapping.tsv).
    * synapse class: class of the synapses of a pre-synaptic neuron. Either 'inhibitory' or
        'excitatory'.
    * layer slice: layers are sliced along the "cortical depth axis", resulting in sublayers of
        equal thickness.
        In "A Derived Positional Mapping of Inhibitory Subtypes in the Somatosensory Cortex",
        our layer slices are called "bins" so as to avoid confusion with actual rat brain slices
        cut orthogonally to the sagital axis. A layer slice in our sense is laminar refinement of
        a layer, orthogonal to the "cortical depth axis".

"""
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from nptyping import NDArray  # type: ignore
from tqdm import tqdm
from voxcell import VoxelData

VoxelIndices = Tuple[NDArray[int], NDArray[int], NDArray[int]]
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
L = logging.getLogger("mtype_densities")


class DensityProfileCollection:
    """
    Class to manage neuron density profiles.

    * load profiles from files
    * assemble a full profile for each specified mtype
    * create a neuron density file (nrrd) for each mtype
    """

    def __init__(
        self,
        mtype_to_profile_map: pd.DataFrame,
        layer_slices_map: Dict[str, range],
        density_profiles: Dict[str, List[float]],
    ) -> None:
        """
        Initialization of DensityProfileCollection

        Args:
            mtype_to_profile_map: dataframe of the following form
                        mtype      synapse_class  layer  profile_name
                    0   L1_DAC     inhibitory     1      L1_DAC
                    1   L1_HAC     inhibitory     1      L1_HAC
                    2   L1_LAC     inhibitory     1      L1_LAC
                    ...

            layer_slices_map: dict of the following form
                {
                    'layer_1': range(93, 100),
                    'layer_2': range(86, 93),
                    'layer_3': range(69, 86),
                    'layer_4': range(60, 69),
                    'layer_5': range(35, 60),
                    'layer_6': range(0, 35),
                }
                This data structure indicates how each layer splits into slices of equal thickness.
                In the above example, there is a total of 100 slices for the whole mouse isocortex.
                Layer 6, for instance, is divided into 35 slices of equal thickness whose indices
                range from 0 to 34.

            stop_layer: name (str) of the layer with the highest slice indices.
                In app/data/meta/layers.tsv, this is 'layer_1' whose slice indices range from 93 to
                99.

            density_profiles: dict of the following form
                {
                    'BP.dat': [0.0, 99.556, 269.83, ...],
                    'BTC.dat': [128.3, 276.48, 439.17, ...],
                    'CHC.dat': [1.82, 33.975, 57.9, ...],
                    'L1_DAC.dat': [0.0, 0.0, 0.0, ...],
                    ...
                }
                The dict keys are names of density profile files and the dict values are lists of
                non-negative float numbers with equal length. The lenght of each list is the total
                number of layer slices defined by `layer_slices_map`. Each list holds the average
                numbers of cells in each slice described in `layer_slices_map`.
        """
        self.mtype_to_profile_map = mtype_to_profile_map
        self.layer_slices_map = layer_slices_map

        # Find the layer with the highest slice indices
        self.stop_layer = max(layer_slices_map, key=lambda key: layer_slices_map[key].stop)

        self.density_profiles = density_profiles
        self.excitatory_mtypes: List[str] = []
        self.inhibitory_mtypes: List[str] = []
        self._collect_density_profiles()

    def _collect_density_profiles(self) -> None:
        """Collect input density profiles and assembles a full profile for each specified mtypes

        This function populates the attribute `self.profile_data` holding the information
        needed to create density nrrd files for each mtype specified in
        `self.mtype_to_profile_map`.

        This function also creates the lists of the excitatory and inhibitory mtypes:
            * self.excitatory_mtypes (e.g., ['L1_DAC', 'L1_HAC', ...])
            * self.inhibitory_mtypes (e.g., ['L2_IPC', 'L2_IPC:A', ...])

        `self.profile_data` is a dict of the form
        {
            'layer_1': {
                'excitatory': <pd.DataFrame>,
                'inhibitory': <pd.DataFrame>,
            },
            'layer_2': {
                'excitatory': <pd.DataFrame>,
                'inhibitory': <pd.DataFrame>,
            },
            ...
        }
        Each dataframe value is either empty or of the form
        (example for layer 4):

                L4_SSC    L4_TPC    L4_UPC
            60  0.090906  0.642046  0.267049
            61  0.090912  0.642041  0.267047
            62  0.090910  0.642044  0.267045
            63  0.090912  0.642043  0.267045
            64  0.090906  0.642048  0.267046
            65  0.090909  0.642044  0.267047
            66  0.090909  0.642048  0.267043
            67  0.090909  0.642047  0.267044
            68  0.090909  0.642043  0.267048

        The row indices are the indices of the layer slices as defined by
        `self.layer_slices_map`. Each column holds the proportion of cells
        in each slice for the corresponding mtype. For instance, around
        9% of the cells in the slice 60 (layer 4) are cells of mtype L4_SSC.
        The sum over each line should be 1.0.
        """
        L.info("Collecting density profiles ...")
        self.profile_data = {
            layer: {"excitatory": pd.DataFrame(), "inhibitory": pd.DataFrame()}
            for layer in self.layer_slices_map
        }
        for _, row in self.mtype_to_profile_map.iterrows():
            for layer_index in row.layer.split(","):  # handle the special case of layer 2/3: '2,3'
                layer = "layer_" + str(layer_index)
                range_ = self.layer_slices_map[layer]
                if row.synapse_class == "excitatory":
                    self.excitatory_mtypes.append(row.mtype)
                elif row.synapse_class == "inhibitory":
                    self.inhibitory_mtypes.append(row.mtype)
                self.profile_data[layer][row.synapse_class][row.mtype] = self.density_profiles[
                    row.profile_name
                ][slice(range_.start, range_.stop)]

        # Set DataDrame index with layer slice indices and normalize rows to
        # get mtype proportions for each layer slice
        for layer, range_ in self.layer_slices_map.items():
            for synapse_class in ["excitatory", "inhibitory"]:
                data_frame = self.profile_data[layer][synapse_class]
                if data_frame.empty:
                    continue
                data_frame.index = range_
                self.profile_data[layer][synapse_class] = data_frame.div(
                    data_frame.sum(axis=1), axis=0
                ).fillna(0.0)
                # Check for each slice if there are cells from at least one mtype
                for row_index, row in self.profile_data[layer][synapse_class].iterrows():
                    if np.sum(row) == 0.0:
                        warnings.warn(
                            f"No {synapse_class} cells assigned to slice {row_index} of {layer}"
                        )

    @classmethod
    def load(
        cls,
        mtype_to_profile_map_path: Union[str, Path],
        layer_slices_path: Union[str, Path],
        density_profiles_dirpath: Union[str, Path],
    ) -> "DensityProfileCollection":
        # fmt: off
        '''
        Load data files, build and return a DensityProfileCollection

        Args:
            mtype_to_profile_map_path: path to the .tsv file describing which neuron density
                profiles should be associated to which mtype.
                The content of such file looks like this (excerpt):
                    mtype	   sclass	layer	file
                    L1_DAC	   INH	    1	    L1_DAC
                    L1_HAC	   INH	    1	    L1_HAC
                    L1_LAC	   INH	    1	    L1_LAC
                    L1_NGC-DA  INH	    1	    L1_NGC-DA
                    ...

            layer_slices_path: path to the .tsv file defining the layer slices
                Each layer is split into several slices of equal thickness.
                Slices are identified by a unique index. The content of such a file
                looks like this:
                    layer	from  upto
                    6 	    0	  35
                    5	    35	  60
                    4	    60	  69
                    3	    69	  86
                    2	    86	  93
                    1	    93	  100
                Here layers and slices are ordered according to cortical depth.

            density_profiles_dirpath: path to the directory containing the neuron density profiles
                under the form of .dat files (e.g., BP.dat, BTC.dat, etc.). Each file contains
                a single column of non-negative float numbers, one cell number for each slice
                defined in `layer_slices_path`.

            Returns:
                DensityProfileCollection object.
        '''
        # fmt: on
        mtype_to_profile_map = pd.read_csv(str(mtype_to_profile_map_path), sep=r"\s+")

        def _get_synapse_class_longname(short_name: str) -> str:
            if short_name == "EXC":
                return "excitatory"
            if short_name == "INH":
                return "inhibitory"
            raise AssertionError(f"Unrecognized synapse class {short_name}")

        L.info("Loading density profiles from files ...")
        mtype_to_profile_map = mtype_to_profile_map.rename(
            columns={"sclass": "synapse_class", "file": "profile_name"}
        )
        mtype_to_profile_map["synapse_class"] = list(
            map(_get_synapse_class_longname, mtype_to_profile_map["synapse_class"])
        )
        # Get a list of profile names without duplicates
        density_profile_filenames = list(dict.fromkeys(mtype_to_profile_map["profile_name"]))
        density_profiles = {
            filename: np.loadtxt(Path(density_profiles_dirpath, filename + ".dat"))
            for filename in density_profile_filenames
        }

        L.info("Loading layer slice ranges from file %s ...", layer_slices_path)
        layer_slices_df = pd.read_csv(str(layer_slices_path), sep=r"\s+")
        layer_slices_map = {
            f"layer_{str(row['layer'])}": range(row["from"], row["upto"])
            for _, row in layer_slices_df.iterrows()
        }

        return cls(mtype_to_profile_map, layer_slices_map, density_profiles)

    @staticmethod
    def load_placement_hints(placement_hints_paths: Dict[str, str]) -> Dict[str, "VoxelData"]:
        """
        Load placement hints nrrd files.

        This can take some time, depending on the volume resolution and the available RAM.

        Args:
            placement_hints_paths: dict whose keys are layer names (str) and whose values
                are paths (str) to the corresponding placement hints nrrd files.
                Example:
                    {
                        'layer_1': '[PH]layer_1.nrrd',
                        'layer_2': '[PH]layer_2.nrrd',
                        ...
                        'y': '[PH]y.nrrd'
                    }

        Returns:
            dict whose keys are layer names and whose values are VoxelData objects holding the
            corresponding layer placement hints array (float array of shape (W, H, D, 2) if
            (W, H, D) is the shape of the underlying annotated volume).

        """
        L.info("Loading placement hints nrrd files ...")
        placement_hints = {
            name: VoxelData.load_nrrd(filepath).raw
            for (name, filepath) in placement_hints_paths.items()
        }
        L.info("Placement hints loaded.")

        return placement_hints

    def compute_layer_slice_voxel_indices(
        self,
        placement_hints_paths: Dict[str, str],
    ) -> Dict[int, VoxelIndices]:
        """
        Compute the voxel indices of each layer slice defined in `self.layer_slices_map`.

        Placement hints (see atlas_building_tools.placement_hints), i.e., distances along direction
        vectors from each voxel to each layer boundary, are used to split each layer in slices of
        roughly equal thickness. The number of slices per layer is determined by
        `self.layer_slices_map`.

        Args:
            placement_hints_paths: dict whose keys are layer names (str) and whose values
                are paths (str) to the corresponding placement hints nrrd files.
                Example:
                    {
                        'layer_1': '[PH]layer_1.nrrd',
                        'layer_2': '[PH]layer_2.nrrd',
                        ...
                        'y': '[PH]y.nrrd'
                    }
        Returns:
            dict whose keys are layer slice indices and whose values are the voxel indices for
            each slice. To each slice index corresponds a tuple (X, Y, Z) whose components are
            arrays of shape (N, ) where N is the number of voxels in the corresponding slice.
        """
        placement_hints = self.load_placement_hints(placement_hints_paths)

        L.info("Computing the voxel indices of each layer slice ...")
        layer_slice_voxel_indices: Dict[int, VoxelIndices] = {}
        phy = placement_hints["y"]
        for layer, range_ in tqdm(self.layer_slices_map.items()):
            phy1 = placement_hints[layer][..., 1]
            phy0 = placement_hints[layer][..., 0]
            layer_thickness = phy1 - phy0
            for i, slice_index in enumerate(range_):
                mask = phy >= phy0 + (i / len(range_)) * layer_thickness
                upper_bound = phy0 + ((i + 1) / len(range_)) * layer_thickness
                # Handle the edge case of the last slice of the stop layer
                if layer == self.stop_layer and slice_index == range_.stop - 1:
                    mask = np.logical_and(mask, phy <= upper_bound)
                else:
                    mask = np.logical_and(mask, phy < upper_bound)
                layer_slice_voxel_indices[slice_index] = np.where(mask)

        return layer_slice_voxel_indices

    def create_density(
        self,
        mtype: str,
        synapse_class: str,
        synapse_class_density: NDArray[float],
        layer_slice_voxel_indices: Dict[int, VoxelIndices],
        output_dirpath: Union[str, Path],
    ) -> None:
        """
        Create and save to file a density field for the specified mtype.

        The density nrrd file is saved in `output_dirpath` under the name `mtype`_density.nrrd.

        Args:
            mtype: the morphological cell type for which the creation of a density nrrd
                file is requested (e.g., L23_DBC, L5_TPC:C or L6_UPC).
            synapse_class: class of the synapses of a pre-synaptic neuron.
                Either 'inhibitory' or 'excitatory'.
            synapse_class_density: volumetric density of the neurons with class
                `synapse_class`: This is a float array of shape (W, H, D) where
                (W, H, D) is the shape of the underlying annotated volume.
            layer_slice_voxel_indices: the list of voxel indices in each layer slice, see
                :meth:`mtype_densities.DensityProfileCollection.compute_layer_slice_voxel_indices`
            output_dirpath: directory path where to save the created density file.
        """
        density = np.zeros_like(synapse_class_density.raw)
        for layer, range_ in self.layer_slices_map.items():
            if mtype in self.profile_data[layer][synapse_class].columns:
                layer_density_profile = self.profile_data[layer][synapse_class][mtype]
                for index in range_:
                    slice_voxel_indices = layer_slice_voxel_indices[index]
                    density[slice_voxel_indices] = (
                        layer_density_profile[index]
                        * synapse_class_density.raw[slice_voxel_indices]
                    )

        synapse_class_density.with_data(density).save_nrrd(
            str(Path(output_dirpath, mtype + "_density.nrrd"))
        )

    def create_mtype_densities(
        self,
        excitatory_neuron_density_path: Union[str, Path],
        inhibitory_neuron_density_path: Union[str, Path],
        placement_hints_config_path: Union[str, Path],
        output_dirpath: Union[str, Path],
    ) -> None:
        """
        Create and save to file a density field for each specified mtype.

        Density nrrd files are saved in `output_dirpath` under the name `mtype`_density.nrrd.

        Args:
            excitatory_neuron_density_path: path to the density nrrd file holding the
                density field of excitatory neurons.
            inhibitory_neuron_density_path: path to the density nrrd file holding the
                density field of inhbitory neurons.
            placement_hints_config_path: path to the yaml file holding the
                the placement hints configuration. Here is an example of yaml content:
                'layerPlacementHintsPaths':
                    'layer_1': '[PH]layer_1.nrrd'
                    'layer_2': '[PH]layer_2.nrrd'
                    'layer_3': '[PH]layer_3.nrrd'
                    'layer_4': '[PH]layer_4.nrrd'
                    'layer_5': '[PH]layer_5.nrrd'
                    'layer_6': '[PH]layer_6.nrrd'
                    'y': '[PH]y.nrrd'
            output_dirpath: directory path where to save the created density files.
        """

        config = yaml.load(open(str(placement_hints_config_path)), Loader=yaml.FullLoader)
        layer_slice_voxel_indices = self.compute_layer_slice_voxel_indices(
            config["layerPlacementHintsPaths"]
        )
        excitatory_neuron_density = VoxelData.load_nrrd(str(excitatory_neuron_density_path))
        inhibitory_neuron_density = VoxelData.load_nrrd(str(inhibitory_neuron_density_path))

        Path(output_dirpath).mkdir(exist_ok=True)

        L.info(
            "Creating density files for the %d excitatory mtypes ...",
            len(self.excitatory_mtypes),
        )
        for mtype in tqdm(self.excitatory_mtypes):
            self.create_density(
                mtype,
                "excitatory",
                excitatory_neuron_density,
                layer_slice_voxel_indices,
                output_dirpath,
            )

        L.info(
            "Creating density files for the %d inhibitory mtype ...",
            len(self.inhibitory_mtypes),
        )
        for mtype in tqdm(self.inhibitory_mtypes):
            self.create_density(
                mtype,
                "inhibitory",
                inhibitory_neuron_density,
                layer_slice_voxel_indices,
                output_dirpath,
            )
