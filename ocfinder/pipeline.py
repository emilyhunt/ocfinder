"""A set of functions for general tasks I like to do. Each one is designed to work on hilariously large scales and
datasets without too many tears."""

import ocelot
import numpy as np
import pickle
import json
import pandas as pd
import gc
import datetime
import matplotlib.pyplot as plt
from scipy import sparse

import warnings

from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional, Callable
from .utilities import print_itertime


class Pipeline(object):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 check_input_shape: Union[bool, Tuple[str], List[str]] = False,
                 required_input_keys: Optional[Union[list, tuple]] = None,
                 required_output_keys: Optional[Union[list, tuple]] = None,
                 ):
        """Superclass for pipeline tasks that can handle all of the boring directory stuff for you.

        Args:
            names (list-like): list of names of fields to run on to save.
            input_dirs (dict of paths or lists of paths): input directories. You have three options:
                - If a path to one file, just one path will be saved
                - If a list of paths, this list will be saved
                - If a path to a directory, then files with a suffix specified by input_patterns[the_dict_key] will be
                  saved.
                All key locations must end up having the same shape if check_input_shape=True, which is specified by the
                requirements of the subclass.
            input_patterns (dict of str): for input_dirs that are a path to a directory, this is the pattern to look
                for. Can just be "*" to look for all files.
                Default: None
            output_dirs (dict of paths): output directories to write to.
            verbose (bool): whether or not to run in verbose mode with informative print statements.
                Default: True
            check_input_shape (bool or list-like): specified by the subclass, this says whether or not all inputs must
                have the same shape. Alternatively, may specify a list of keys not to check.
                Default: False

        """
        self.verbose = verbose
        if self.verbose:
            print("Pipeline superclass is grabbing all required directories and checking your input!")

        # Check that the right number of keys exists
        self._check_keys(required_input_keys, input_dirs)
        self.has_output = self._check_keys(required_output_keys, output_dirs)

        self.input_paths = {}
        self.output_paths = {}
        self.names = names

        if check_input_shape is True:
            inputs_not_to_check = []
        elif check_input_shape is False:
            inputs_not_to_check = required_input_keys
        else:
            inputs_not_to_check = check_input_shape

        # Cycle over the input options
        first_length = None
        for a_key in input_dirs.keys():

            # List of inputs
            if type(input_dirs[a_key]) == list:
                print("list")
                self.input_paths[a_key] = input_dirs[a_key]

            # Input is a directory
            elif input_dirs[a_key].is_dir():
                self.input_paths[a_key] = list(Path(input_dirs[a_key]).glob(input_patterns[a_key]))
                self.input_paths[a_key].sort()

            # Just one input
            elif input_dirs[a_key].exists():
                self.input_paths[a_key] = [input_dirs[a_key]]

            # Raise error
            else:
                raise ValueError(f"the input for key '{a_key}' is not a list of paths, a directory or a singular path.")

            # Value checking
            if a_key not in inputs_not_to_check:
                if first_length is None:
                    first_length = len(self.input_paths[a_key])
                else:
                    if len(self.input_paths[a_key]) != first_length:
                        raise ValueError(f"the length of input_dir type {a_key} does not match the length of the first "
                                         f"key! All input keys must have the same length.")

        # Cycle over the output options, making paths where necessary
        if self.has_output:
            for a_key in output_dirs.keys():
                # If it's just one file then we wanna keep it around
                if output_dirs[a_key].is_file():
                    self.output_paths[a_key] = output_dirs[a_key]

                # Otherwise, for dirs, we check that it exists, make it if not, and save it
                else:
                    output_dirs[a_key].mkdir(parents=True, exist_ok=True)
                    self.output_paths[a_key] = output_dirs[a_key]

    @staticmethod
    def _check_keys(required: Optional[Union[list, tuple]], input_dict: dict):
        """Checks that an input dictionary has the correct keys and the correct number of keys.

        Args:
            required (list of str or optional): required keys for input_dict. If None, then assumed no keys needed.
            input_dict (dict): a dict that should have dict.keys() == required.

        Returns:
            Whether or not there are any required keys: True if required is not None, or False if it is
        """
        if required is not None:
            if input_dict is None:
                raise ValueError(f"class requires that paths {required} are specified, but none were!")

            if not all(k in input_dict.keys() for k in required):
                raise ValueError(f"a dictionary of paths does not have the required keys! \n"
                                 f"required keys: {required} \n"
                                 f"actual keys: {input_dict.keys()}")

            # Old code version that required the input keys were *all* shared with the required ones:
            # if len(input_dict) != len(required) or not all(k in required for k in input_dict.keys()):
            #     raise ValueError(f"a dictionary of paths does not have the required number of/names of keys! \n"
            #                      f"required keys: {required} \n"
            #                      f"actual keys: {input_dict.keys()}")

            return True

        else:
            return False

    @staticmethod
    def _open_multiple_files(paths: Union[list, tuple, np.ndarray]):
        """Wrapper for _open that opens a list of paths at once and combines them, intended for use with tables that
        were split for memory reasons. All tables must have the same column names and the same file extension!

        Supports:
        .csv
        .feather

        """
        mode = paths[0].suffix

        if mode == '.csv':
            readfunc = pd.read_csv
        elif mode == '.feather':
            readfunc = pd.read_feather
        else:
            raise ValueError(f"Specified file type {mode} not supported!")

        files = [None] * len(paths)
        for i, a_path in enumerate(paths):
            files[i] = readfunc(a_path)

        return pd.concat(files, ignore_index=True)

    @staticmethod
    def _open(path: Path):
        """Cheeky function to read in Gaia data from a pickle file, JSON, etc...

        Supports:
        .pickle
        .json
        .jsonpanda (reads straight to a pandas object)
        .csv (also reads straight to a pandas object)
        .npz
        .npy
        .feather
        """
        mode = path.suffix

        if mode == '.pickle':
            with open(str(path.resolve()), 'rb') as handle:
                return pickle.load(handle)
        elif mode == '.json':
            with open(str(path.resolve()), 'r') as handle:
                return json.load(handle)
        elif mode == '.jsonpanda':
            return pd.read_json(path)
        elif mode == '.csv':
            return pd.read_csv(path)
        elif mode == '.feather':
            return pd.read_feather(path)
        elif mode == '.npy':
            return np.load(str(path.resolve()))
        elif mode == '.npz':
            numpy_files = np.load(str(path.resolve()))

            # Deal with if it's actually a sparse matrix
            if numpy_files.files == ['indices', 'indptr', 'format', 'shape', 'data']:
                del numpy_files
                return sparse.load_npz(str(path.resolve()))

            # Raise an error if there's more than one file here
            elif len(numpy_files.files) != 1:
                raise NotImplementedError(f"Only loading of singular arrays is supported, but the .npz file passed "
                                          f"at {path} has 2+ or 0 arrays.")

            # Or, just return our friend
            else:
                return numpy_files[numpy_files.files[0]]
        else:
            raise ValueError(f"Specified file type {mode} not supported!")


# A blank statistics DataFrame. We use this in case no clusters are returned so that a blank cluster list can be saved
# instead for that field.
_blank_statistics_dataframe = pd.DataFrame(columns=[
    'field', 'run', 'cluster_label', 'cluster_id', 'n_stars',
    'ra', 'ra_error', 'dec', 'dec_error', 'ang_radius_50',
    'ang_radius_50_error', 'ang_radius_c', 'ang_radius_c_error',
    'ang_radius_t', 'ang_radius_t_error', 'radius_50', 'radius_50_error',
    'radius_c', 'radius_c_error', 'radius_t', 'radius_t_error', 'parallax',
    'parallax_error', 'inverse_parallax', 'inverse_parallax_l68',
    'inverse_parallax_u68', 'distance', 'distance_error', 'pmra',
    'pmra_error', 'pmdec', 'pmdec_error', 'v_internal_tangential',
    'v_internal_tangential_error', 'parameter_inference_mode',
])


class ClusteringAlgorithm(Pipeline):
    def __init__(self,
                 algorithm: Callable,
                 user_kwargs: Optional[Union[List[dict], Tuple[dict]]],
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 required_input_keys: Union[list, tuple] = ('data', 'rescaled'),
                 required_output_keys: Union[list, tuple] = ('data', 'labels', 'times'),
                 extra_returned_info: Union[list, tuple] = (),
                 max_cluster_size_to_save: int = 10000, ):
        """Intermediary superclass between the Pipeline class and clustering algorithms. Implements apply(),
        save_clustering_result() and save_clustering_time() methods.

        Args:
            algorithm (function): callable algorithm that returns at least labels, probabilities.
                E.g. ocfinder.hdbscan.run_hdbscan()
            names (list-like): list of names of fields to run on to save.
            input_dirs (dict of paths or lists of paths): input directories. You have three options:
                - If a path to one file, just one path will be saved
                - If a list of paths, this list will be saved
                - If a path to a directory, then files with a suffix specified by input_patterns[the_dict_key] will be
                  saved.
                All key locations must end up having the same shape if check_input_shape=True, which is specified by the
                requirements of the subclass.
            input_patterns (dict of str): for input_dirs that are a path to a directory, this is the pattern to look
                for. Can just be "*" to look for all files.
                Default: None
            output_dirs (dict of paths): output directories to write to.
            verbose (bool): whether or not to run in verbose mode with informative print statements.
                Default: True
            required_input_keys (list, tuple, optional): required input keys for the algorithm. This is pretty much
                always going to be ('data', 'rescaled') for every algorithm and shouldn't need changing.
                Default: ('data', 'rescaled')
            required_output_keys (list, tuple, optional): required output keys for saving. For clustering algorithms,
                they should be one of:
                - 'data'  (for data_gaia views per-cluster)
                - 'labels'
                - 'probabilities'
                - 'cluster' (for wherever per-cluster info will be left)
                - 'times' (for wherever recorded runtimes will be left)
                Default: ('data', 'labels', 'times')  (a simple combo)
            extra_returned_info (list, tuple): names of extra info columns returned by algorithm. If specified and if
                a 'cluster' Path in required_output_keys/output_dirs is specified, then these columns will be saved
                to the per-cluster info dataframe.
            max_cluster_size_to_save (int): maximum cluster size to save baby_data_gaia views for. If 0 or -ve, then we
                also don't calculate any statistics for the cluster, and just return the labels (+probabilities).
                Default: 10000

        """

        super().__init__(names, input_dirs, input_patterns=input_patterns, output_dirs=output_dirs, verbose=verbose,
                         check_input_shape=['times'],
                         required_input_keys=required_input_keys,
                         required_output_keys=required_output_keys)

        if self.verbose:
            print("  initiating ClusteringAlgorithm intermediary superclass")

        self.algorithm = algorithm
        self.algorithm_name = algorithm.__name__

        # Handle the algorithm kwargs if the user just wants to use the defaults
        if user_kwargs is None:
            self.n_kwarg_sets = 1
            self.algorithm_kwargs = [{}]

        # Otherwise, if more than one set is specified, then we'll loop over
        else:
            self.n_kwarg_sets = len(user_kwargs)
            self.algorithm_kwargs = user_kwargs

        # Remember whether or not an extra input is needed
        if len(required_input_keys) == 3:
            self.third_input_name = required_input_keys[2]
        else:
            self.third_input_name = None

        # Infer what kind of output the algorithm has
        self._save_probabilities = 'probabilities' in required_output_keys
        self._save_data_gaia_views = 'cluster_data' in required_output_keys
        self._n_extra_info = len(extra_returned_info)  # Whether the algorithm returns anything extra
        self._extra_info_names = extra_returned_info

        # We save a cluster array if there's required output key info or extra info incoming.
        # The cluster dataframe is individual information on each cluster, which can include:
        # - Temporary IDs
        # - Calculated statistics
        # - Extra information from the algorithm (e.g. means/stds from GMMs)
        self._save_cluster_info = 'cluster_list' in required_output_keys
        self._calculate_cluster_stats = max_cluster_size_to_save > 0
        self._max_cluster_size_for_stats = max_cluster_size_to_save

        if self.verbose:
            print("  initialisation is complete! Hurrah \\o/")

    def _get_clusterer_args(self, path: Path, input_number):
        """Cheeky hack that lets me use an extra input argument if necessary on certain functions (like GMMs)."""
        # Feather rescaled files have useless column names which we drop now
        if path.suffix == '.feather':
            main_data = self._open(path).values
        # Otherwise, we keep the file (e.g. if it's a scipy csr sparse matrix)
        else:
            main_data = self._open(path)

        if self.third_input_name is None:
            return [main_data]
        else:
            third_data = self._open(self.input_paths[self.third_input_name][input_number])
            return [main_data, third_data]

    def apply(self, start: int = 0):
        """Applies an algorithm to the pre-specified clusters.

        Args:
            start (int): the cluster number in self.names to start with.
                Default: 0

        """
        completed_steps = start
        total_steps = len(self.input_paths['cut']) * self.n_kwarg_sets

        iteration_start = datetime.datetime.now()

        # Loop over all the different fields
        for input_number, a_field_name, in enumerate(self.names[start:], start=start):

            a_data_gaia_path = self.input_paths['cut'][input_number]
            a_data_rescaled_path = self.input_paths['rescaled'][input_number]

            # Also loop over all of the different specified parameter sets
            for i_parameter_set, an_algorithm_kwargs in enumerate(self.algorithm_kwargs, start=1):

                if self.verbose:
                    print(f"-- {datetime.datetime.today()}")
                    print(f"Running {self.algorithm_name} on cluster {a_field_name}!")
                    print(f"  parameter set = {i_parameter_set} of {self.n_kwarg_sets}")

                algorithm_args = self._get_clusterer_args(a_data_rescaled_path, input_number)

                start = datetime.datetime.today()
                algorithm_return = self.algorithm(*algorithm_args, **an_algorithm_kwargs)
                runtime = (datetime.datetime.today() - start).total_seconds()

                # Am I paranoid?
                del algorithm_args
                gc.collect()

                if self.verbose:
                    print(f"  successfully ran in {runtime}s! Saving the output...")

                data_gaia = self._open(a_data_gaia_path)
                n_clusters = self._save_clustering_result(data_gaia, algorithm_return, a_field_name, i_parameter_set)
                self._save_clustering_time(a_field_name, i_parameter_set, runtime, n_clusters, an_algorithm_kwargs)

                # You bet I am.
                del data_gaia, algorithm_return
                gc.collect()

                completed_steps += 1
                if self.verbose:
                    print_itertime(iteration_start, completed_steps, total_steps)

    def _save_clustering_result(self,
                                data_gaia: pd.DataFrame,
                                algorithm_return: Union[list, tuple],
                                field_name: Union[str, int],
                                run_name: Union[str, int]):
        """Saves the result of a clustering analysis.

        Args:
            data_gaia (pd.DataFrame): Gaia data. You know the drill by now.
            algorithm_return (list, tuple): return from the algorithm. At the very least should have length 2, where
                the first element is the labels and the second is the probabilities or None.
            name (str): field_name of the clustering result currently being studied, e.g. "0042_dbscan_acg"

        Returns:
            n_clusters (int): the number of clusters found.

        """
        joined_name = f"{field_name}_{run_name}"

        # De-compose the returned list
        labels = algorithm_return[0]
        probabilities = algorithm_return[1]

        # Save the labels and probabilities arrays
        pd.DataFrame({'labels': labels}).to_feather(
            self.output_paths['labels'] / Path(f"{joined_name}_labels.feather"))

        if self._save_probabilities:
            pd.DataFrame({'probabilities': probabilities}).to_feather(
                self.output_paths['probabilities'] / Path(f"{joined_name}_probs.feather"))

        # Grab stuff we need
        all_cluster_indices = np.unique(labels)
        all_cluster_indices = all_cluster_indices[all_cluster_indices != -1]
        n_clusters = len(all_cluster_indices)

        # Also save either mini views of data_gaia or cluster info
        if self._save_data_gaia_views or self._save_cluster_info:

            parameter_frames_to_concatenate = []

            # Run over all the clusters
            for a_label in all_cluster_indices:
                # Name the cluster!!!!
                cluster_id = joined_name + f"_{a_label}"

                # Calculate all statistics with ocelot
                a_dict = {'field': field_name,
                          'run': run_name,
                          'cluster_label': a_label,
                          'cluster_id': cluster_id}

                # Make a baby data gaia if it's needed
                if self._save_data_gaia_views or self._calculate_cluster_stats:
                    good_stars = labels == a_label
                    baby_data_gaia = data_gaia.loc[good_stars].reset_index(drop=True)

                    if self._save_probabilities:
                        baby_probabilities = probabilities[good_stars]
                    else:
                        baby_probabilities = None

                    # Calculate cluster stats
                    if self._calculate_cluster_stats:
                        a_dict.update(ocelot.calculate.all_statistics(
                            baby_data_gaia, membership_probabilities=baby_probabilities))

                    # Save the small data_gaia view
                    if self._save_data_gaia_views and a_dict['n_stars'] < self._max_cluster_size_for_stats:

                        if self._save_probabilities:
                            baby_data_gaia['probability'] = baby_probabilities

                        baby_data_gaia.to_csv(
                            self.output_paths['cluster_data'] / Path(f"{cluster_id}_cluster_data.csv"))

                    del baby_data_gaia
                    gc.collect()

                parameter_frames_to_concatenate.append(pd.DataFrame(a_dict, index=[0]))

            # Concatenate into a DataFrame
            if len(parameter_frames_to_concatenate) > 0:
                final_cluster_info = pd.concat(parameter_frames_to_concatenate, ignore_index=True)
            else:
                final_cluster_info = _blank_statistics_dataframe.copy()

            # Add extra info if desired
            if self._n_extra_info > 0:

                extra_columns_for_cluster_info = {}

                for index, an_extra_column in enumerate(self._extra_info_names, start=2):
                    extra_columns_for_cluster_info[an_extra_column] = algorithm_return[index]

                extra_columns_for_cluster_info = pd.DataFrame(extra_columns_for_cluster_info)
                final_cluster_info = final_cluster_info.join(extra_columns_for_cluster_info)

            # Concatenate into a dataframe!
            if self._save_cluster_info:
                final_cluster_info.to_csv(self.output_paths['cluster_list'] / Path(f"{joined_name}_cluster_list.csv"),
                                          index=False)

        return n_clusters

    def _save_clustering_time(self,
                              field_name: Union[str, int],
                              run_name: Union[str, int],
                              time: float,
                              n_clusters: int,
                              parameters: dict,):
        """Saves the timing results of a clustering analysis. Makes a new csv at output_paths['times'] or writes a new
        one.

        Args:
            field_name (str or int): name of the field to save the result for.
            run_name (str or int): name of the run to save the result for.
            time (float): times of the field to save the result for.
            n_clusters (int): the number of clusters found.
            parameters (float): kwargs passed to the clustering algorithm at runtime.

        """
        # Don't over-write (just append) if the file already exists
        times_filename = self.output_paths['times'] / Path('runtimes.csv')
        if times_filename.exists():
            kwargs = {'mode': 'a', 'header': False}
        else:
            kwargs = {}

        pd.DataFrame({'field_name': field_name,
                      'run_name': run_name,
                      'algorithm': self.algorithm_name,
                      'time': time,
                      'n_clusters': n_clusters,
                      'parameters': str(parameters)},
                     index=[0]).to_csv(times_filename, index=False, **kwargs)


default_rescale_kwargs = {
    'columns_to_rescale': ('lon', 'lat', 'pmlon', 'pmlat', 'parallax'),
    'column_weights': (1., 1., 1., 1., 1.),
    'scaling_type': 'robust',
    'concatenate': True,
    'return_scaler': True,
}


class Preprocessor(Pipeline):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, list]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 cuts: dict = None,
                 centers: Optional[Union[list, tuple, np.ndarray]] = None,
                 pixel_ids: Optional[Union[list, tuple, np.ndarray]] = None,
                 split_files: bool = False,
                 **kwargs_for_rescaler):
        """Pre-processor for taking Gaia data as input and calculating all required stuff.

        Args:
            names (list-like): list of names of fields to run on to save.
            input_dirs (dict of paths or lists of paths): input directories. You have three options:
                - If a path to one file, just one path will be saved
                - If a list of paths, this list will be saved
                - If a path to a directory, then files with a suffix specified by input_patterns[the_dict_key] will be
                  saved.
                Must specify 'data'.
            input_patterns (dict of str): for input_dirs that are a path to a directory, this is the pattern to look
                for. Can just be "*" to look for all files.
                Default: None
            output_dirs (dict of paths): output directories to write to. Must specify 'cut', 'scaler' and 'rescaled'.
            verbose (bool): whether or not to run in verbose mode with informative print statements.
                Default: True
            cuts (dict): cuts to pass to ocelot.preprocess.cut_dataset.
                Default: None
            centers (list, tuple, np.ndarray): centers of the fields to re-center, with shape (n_fields, 2) and format
                (ra, dec). Must specify me or pixel_ids.
                Default: None
            pixel_ids (list, tuple, np.ndarray): pixel ids of the central pixel in every field, with shape (n_fields,).
                Must specify me or centers.
                Default: None
            split_files (bool): whether or not files are split into multiple .csv files. This can be the case for
                Gaia data downloads of separate HEALPix pixels. All files with the same field name will hence be merged.
                Default: False
            **kwargs_for_rescaler: additional kwargs to pass to the rescaler/defaults to over-write.

        """
        super().__init__(names,
                         input_dirs,
                         input_patterns=input_patterns, output_dirs=output_dirs, verbose=verbose,
                         required_input_keys=['data'],
                         required_output_keys=['cut', 'scaler', 'rescaled'],
                         check_input_shape=True)

        if cuts is None:
            cuts = {}
        self.cuts = cuts

        if centers is not None and pixel_ids is None:
            self.center_mode = True
            self.centers = centers
        elif pixel_ids is not None and centers is None:
            self.center_mode = False
            self.centers = pixel_ids

        self.rescale_kwargs = default_rescale_kwargs
        self.rescale_kwargs.update(**kwargs_for_rescaler)
        self.split_files = split_files

        if self.split_files:
            self._handle_split_files()

        if self.verbose:
            print("Preprocessing pipeline initialised!")

    def _handle_split_files(self):
        """Handles files being split into multiple components/pixels. This can be the case for individual download of
        Gaia HEALPix cells. Currently, will join everything with the same id into lists of paths to each file."""

        # Get the field ID of all files, which should be the part of the file stem before the first underscore
        warnings.warn("split files in the preprocessor are an experimental feature. Proceed with caution! "
                      "Make sure that all file names have a field id at the start of the file stem that does NOT "
                      "contain underscores.")
        field_ids = [x.stem.rsplit(sep="_")[0] for x in self.input_paths['data']]
        unique_field_ids, counts = np.unique(field_ids, return_counts=True)

        # Now, let's use numpy vectorisation to select all fields in the sets of matches and make them into
        vector_input_paths = np.asarray(self.input_paths['data'], dtype=object)
        matches = unique_field_ids.reshape(-1, 1) == field_ids
        self.input_paths['data'] = [vector_input_paths[a_match_set].tolist() for a_match_set in matches]

        print(self.input_paths['data'])

    def apply(self):
        """Applies the pre-processor."""
        completed_steps = 0
        total_steps = len(self.names)

        iteration_start = datetime.datetime.now()

        # Cycle over each cluster, applying all of the required pre-processing steps
        for a_path, a_center, a_name in zip(self.input_paths['data'], self.centers, self.names):

            if self.verbose:
                print(f"Working on field {a_name}...")
                print(f"  cutting dataset")

            if self.split_files:
                data_gaia = self._open_multiple_files(a_path)
            else:
                data_gaia = self._open(a_path)

            # Apply cuts and scaling
            data_gaia = ocelot.cluster.cut_dataset(data_gaia, self.cuts)

            if self.verbose:
                print(f"  re-centering dataset")
            if self.center_mode:
                data_gaia = ocelot.cluster.recenter_dataset(data_gaia, center=a_center)
            else:
                data_gaia = ocelot.cluster.recenter_dataset(data_gaia, pixel_id=a_center, rotate_frame=True)

            if self.verbose:
                print(f"  re-scaling dataset")
            data_rescaled, scaler = ocelot.cluster.rescale_dataset(data_gaia, **self.rescale_kwargs)
            data_rescaled = pd.DataFrame(data_rescaled, columns=[str(x) for x in range(data_rescaled.shape[1])])

            if self.verbose:
                print(f"  saving them both!")

            data_gaia.to_feather(self.output_paths['cut'] / Path(f"{a_name}_cut.feather"))

            data_rescaled.to_feather(self.output_paths['rescaled'] / Path(f"{a_name}_rescaled.feather"))

            with open(self.output_paths['scaler'] / Path(f"{a_name}_scaler.pickle"), 'wb') as handle:
                pickle.dump(scaler, handle, pickle.HIGHEST_PROTOCOL)

            del data_gaia, scaler, data_rescaled
            gc.collect()

            if self.verbose:
                print(f"  both saved successfully!")

            completed_steps += 1
            if self.verbose:
                print_itertime(iteration_start, completed_steps, total_steps)

        if self.verbose:
            print(f"  pre-processing is complete!")


default_result_plotting_kwargs = {
    'show_figure': False
}


class ResultPlotter(Pipeline):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 **kwargs_for_plotting_algorithm):
        """Pre-processor for taking Gaia data as input and calculating all required stuff.

        Args:
            names (list-like): list of names of fields to run on to save.
            input_dirs (dict of paths or lists of paths): input directories. You have three options:
                - If a path to one file, just one path will be saved
                - If a list of paths, this list will be saved
                - If a path to a directory, then files with a suffix specified by input_patterns[the_dict_key] will be
                  saved.
                Must specify 'cut', 'labels', 'cluster_list', 'times'. May optionally specify 'probabilities'.
            input_patterns (dict of str): for input_dirs that are a path to a directory, this is the pattern to look
                for. Can just be "*" to look for all files.
                Default: None
            output_dirs (dict of paths): output directories to write to. Must specify 'plots'.
            verbose (bool): whether or not to run in verbose mode with informative print statements.
                Default: True
            **kwargs_for_plotting_algorithm: additional kwargs to pass to the plotting function/defaults to over-write.

        """
        super().__init__(names,
                         input_dirs,
                         input_patterns=input_patterns, output_dirs=output_dirs, verbose=verbose,
                         required_input_keys=['cut', 'labels', 'cluster_list', 'times'],  # Can also be 'probabilities'
                         required_output_keys=['plots'],
                         check_input_shape=['cut', 'times'])

        if len(self.input_paths['times']) != 1:
            raise ValueError("multiple runtime metadata files were specified, but only one big one can be used by "
                             "this function.")

        self.plot_kwargs = default_result_plotting_kwargs
        self.plot_kwargs.update(**kwargs_for_plotting_algorithm)
        self.first_generate_figure_title_call = True
        self.base_figure_title = None
        self.base_figure_savename = None

        # Has the user specified probabilities to plot?
        self.plot_probabilities = 'probabilities' in input_dirs.keys()

        # Get a list of everything we'll need to plot
        # N.B. THIS WILL NOT WORK if the run_name contains underscores! File names are assumed to look like:
        # <field_name>_<run_name>_labels
        self.cut_data_field_names = np.asarray(
            ["_".join(x.stem.rsplit(sep="_")[:-1]) for x in self.input_paths['cut']], dtype=str)

        self.all_label_file_stems = [x.stem for x in self.input_paths['labels']]

        self.field_names = [''] * len(self.all_label_file_stems)
        self.run_names = [''] * len(self.all_label_file_stems)

        for i, a_file_stem in enumerate(self.all_label_file_stems):
            self.field_names[i], self.run_names[i] = self._get_field_and_run_name_from_file_name(a_file_stem)

        # Cast as numpy arrays to make later stuff work
        self.field_names = np.asarray(self.field_names)
        self.run_names = np.asarray(self.run_names)

        if self.verbose:
            print("Plotting pipeline initialised!")

    @staticmethod
    def _get_field_and_run_name_from_file_name(file_stem: str, has_run_name: bool = True):
        if has_run_name:
            split_up = file_stem.rsplit(sep="_")
            field_name = "_".join(split_up[:-2])
            run_name = split_up[-2]
            return field_name, run_name

        else:
            split_up = file_stem.rsplit(sep="_")
            field_name = "_".join(split_up[:-1])
            return field_name

    @staticmethod
    def _check_matches(matches: np.ndarray, name: str, desired_number: int = 1):
        """Checks that the correct number of file matches has been found and raises an informative error if not."""
        n_matches = len(matches)
        if n_matches != desired_number:
            raise ValueError(f"{n_matches} files were found for {name}, but only {desired_number} should exist. "
                             f"Check that the input file list does not contain duplicates and that no run_names contain"
                             f" underscores.")

    def _get_plotting_files(self,
                            index_of_labels_file: int,
                            field_name: str,
                            run_name: str,
                            data_runs: pd.DataFrame,
                            open_cut_data: bool):
        """Returns files to use for a given field_name and run_name."""
        # First off, let's find the cut data
        if open_cut_data:
            cut_data_matches = (self.cut_data_field_names == field_name).nonzero()[0]
            self._check_matches(cut_data_matches, f"field {field_name} cut_data")
            cut_data = self._open(self.input_paths['cut'][cut_data_matches[0]])
        else:
            cut_data = None

        # The next few things are a bit easier as the labels array is required to have the same shape as all of its
        # other friends. We do more checking than is strictly necessary
        labels_file = self.input_paths['labels'][index_of_labels_file]
        labels_stem = "_".join(labels_file.stem.rsplit(sep="_")[:-1])

        cluster_list_file = self.input_paths['cluster_list'][index_of_labels_file]
        cluster_list_file_stem = "_".join(cluster_list_file.stem.rsplit(sep="_")[:-2])

        if self.plot_probabilities:
            probabilities_file = self.input_paths['probabilities'][index_of_labels_file]
            probabilities_file_stem = "_".join(probabilities_file.stem.rsplit(sep="_")[:-1])
        else:
            probabilities_file = None
            probabilities_file_stem = labels_stem

        if labels_stem != cluster_list_file_stem:
            raise ValueError("input labels and cluster list files are not in the same order!")
        if labels_stem != probabilities_file_stem:
            raise ValueError("input labels and probabilities files are not in the same order!")
        if labels_stem != f"{field_name}_{run_name}":
            raise RuntimeError("something has gone wrong between __init__ and now, because the labels file here is "
                               "not the same as the one specified by field_name and run_name in the call of this func.")

        # Open it all!
        labels = self._open(labels_file)['labels'].values
        cluster_list = self._open(cluster_list_file)
        if self.plot_probabilities:
            probabilities = self._open(probabilities_file)['probabilities'].values
        else:
            probabilities = None

        # Lastly, let's find whichever row in data_runs is the one we want
        data_runs_matches = np.logical_and(data_runs['field_name'].to_numpy(dtype=str) == field_name,
                                           data_runs['run_name'].to_numpy(dtype=str) == run_name).nonzero()[0]
        self._check_matches(data_runs_matches, f"field {field_name} run {run_name} run metadata (times)")
        series_runs = data_runs.loc[data_runs_matches[0]]

        if open_cut_data:
            return cut_data, labels, probabilities, cluster_list, series_runs
        else:
            return labels, probabilities, cluster_list, series_runs

    @staticmethod
    def _get_threshold_passes(series: pd.Series, threshold: Union[int, float], threshold_comparison: str):
        if threshold_comparison == '>':
            return series.to_numpy() > threshold
        elif threshold_comparison == '<':
            return series.to_numpy() < threshold
        if threshold_comparison == '>=':
            return series.to_numpy() >= threshold
        elif threshold_comparison == '<=':
            return series.to_numpy() <= threshold
        elif threshold_comparison == '==':
            return series.to_numpy() == threshold
        else:
            raise ValueError("unsupported threshold_comparison specified: must be one of '>', '<', '>=', '<=', '=='.")

    def _generate_figure_title(self, cluster_list, series_runs, clusters_to_plot: Union[str, int] = 'all'):
        # If there's already a figure title in the kwargs for the algorithm, then we want to use that friend
        if self.first_generate_figure_title_call:
            if 'figure_title' in self.plot_kwargs.keys():
                self.base_figure_title = str(self.plot_kwargs['figure_title']) + '\n'
            else:
                self.base_figure_title = ''

            if 'save_name' in self.plot_kwargs.keys():
                self.base_figure_savename = str(self.plot_kwargs['save_name']) + '_'
            else:
                self.base_figure_savename = ''

            self.first_generate_figure_title_call = False

        if clusters_to_plot != 'all':
            savename_suffix = '_' + str(clusters_to_plot)
        else:
            savename_suffix = ''

        # Now, let's make us a figure title of fun and happiness!!!
        self.plot_kwargs['figure_title'] = (
            self.base_figure_title +
            f"Clustering analysis with {series_runs['algorithm']}, "
            f"field: {series_runs['field_name']}, run: {series_runs['run_name']}\n"
            f"total clusters (num valid): {series_runs['n_clusters']}  ({cluster_list.shape[0]}), "
            f"plotted clusters: {clusters_to_plot}\n"
            f"non-default parameters: {series_runs['parameters']}\n"
            f"runtime: {series_runs['time']:.2f}s  ({series_runs['time'] / 60**2:.2f}h)"
        )

        # And a save name too =)
        self.plot_kwargs['save_name'] = (self.output_paths['plots'] / Path(
            self.base_figure_savename
            + f"{series_runs['algorithm']}_{series_runs['field_name']}_{series_runs['run_name']}{savename_suffix}.png"))

    def apply(self,
              plot_clusters_individually: bool = False,
              threshold: Optional[Union[int, float]] = None,
              threshold_key: Optional[str] = None,
              threshold_comparison: str = '>'):
        """Runs the plotting pipeline on lots of label arrays, generating plots with ocelot.plot.clustering_result.

        Args:
            plot_clusters_individually (bool): whether or not to make separate plots for every cluster.
                Default: False
            threshold (float, int, optional): a plotting threshold to ignore clusters below.
                Default: None
            threshold_key (str, optional): if threshold specified, this is the key into the cluster_list to look for
                it. Must be specified if threshold is specified!
                Default: None
            threshold_comparison (str): operator to use for comparison.
                Default: '>', i.e. the value found by threshold_key must be greater than threshold.

        """
        if (threshold is not None and threshold_key is None) or (threshold_key is not None and threshold is None):
            raise ValueError("if threshold is specified then threshold_key must be specified too!")

        # Get the runtime metadata file
        data_runs = self._open(self.input_paths['times'][0])

        # Cycle over fields and runs
        last_field_name = 'if you get a match with this string, then honestly you win a prize'
        data_cut = None

        completed_steps = 0
        total_steps = len(self.field_names)
        iteration_start = datetime.datetime.now()

        for i_labels_file, (a_field_name, a_run_name) in enumerate(zip(self.field_names, self.run_names)):

            if self.verbose:
                print(f"-- {datetime.datetime.today()}")
                print(f"Plotting field {a_field_name} on run {a_run_name}!")

            # Grab everything
            if a_field_name != last_field_name:
                del data_cut
                data_cut, labels, probabilities, cluster_list, series_runs = self._get_plotting_files(
                    i_labels_file, a_field_name, a_run_name, data_runs, open_cut_data=True)
            else:
                labels, probabilities, cluster_list, series_runs = self._get_plotting_files(
                    i_labels_file, a_field_name, a_run_name, data_runs, open_cut_data=False)

            if self.verbose:
                print(f"  {series_runs['algorithm']} found {series_runs['n_clusters']} clusters in total")

            # Work out which clusters to plot if there's a threshold
            if threshold is not None:
                good_clusters = self._get_threshold_passes(cluster_list[threshold_key], threshold, threshold_comparison)
                cluster_indices_to_plot = cluster_list.loc[good_clusters, 'cluster_label'].to_numpy()
            else:
                cluster_indices_to_plot = cluster_list['cluster_label'].to_numpy()

            if self.verbose:
                print(f"  found {len(cluster_indices_to_plot)} valid clusters to plot")

            # Individual cluster plotting!
            if plot_clusters_individually:
                for a_cluster_label in cluster_indices_to_plot:

                    if self.verbose:
                        print(f"  making individual plot for cluster {a_cluster_label}...")

                    self._generate_figure_title(cluster_list, series_runs, clusters_to_plot=a_cluster_label)

                    ocelot.plot.clustering_result(
                        data_cut, labels, [a_cluster_label], probabilities, **self.plot_kwargs)

                    if not self.plot_kwargs['show_figure']:
                        plt.close('all')

            # Plot the whole field!
            if self.verbose:
                print(f"  making plot for whole field...")

            self._generate_figure_title(cluster_list, series_runs, clusters_to_plot='all')
            ocelot.plot.clustering_result(data_cut, labels, cluster_indices_to_plot, probabilities, **self.plot_kwargs)

            if not self.plot_kwargs['show_figure']:
                plt.close('all')

            # And now, wrap things up around here
            del labels, probabilities, cluster_list, series_runs
            gc.collect()

            last_field_name = a_field_name

            completed_steps += 1
            if self.verbose:
                print_itertime(iteration_start, completed_steps, total_steps)
