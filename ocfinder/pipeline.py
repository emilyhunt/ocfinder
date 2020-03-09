"""A set of functions for general tasks I like to do. Each one is designed to work on hilariously large scales and
datasets without too many tears."""

import ocelot
import numpy as np
import pickle
import json
import pandas as pd
import gc
import datetime

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
                 check_input_shape: bool = False,
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
            check_input_shape (bool): specified by the subclass, this says whether or not all inputs must have the same
                shape. Default: False

        """
        self.verbose = verbose
        if self.verbose:
            print("Pipeline superclass is grabbing all required directories and checking your input!")

        # Check that the right number of keys exists
        self._check_keys(required_input_keys, input_dirs)
        self._check_keys(required_output_keys, output_dirs)

        self.input_paths = {}
        self.output_paths = {}
        self.names = names

        # Cycle over the input options
        first_length = None
        for a_key in input_dirs.keys():

            # List of inputs
            if type(input_dirs[a_key]) == list:
                self.input_paths[a_key] = input_dirs[a_key]

            # Input is a directory
            elif input_dirs[a_key].is_dir():
                self.input_paths[a_key] = list(Path(input_dirs[a_key]).glob(input_patterns[a_key]))

            else:
                self.input_paths[a_key] = [input_dirs[a_key]]

            # Value checking
            if check_input_shape and first_length is None:
                first_length = len(self.input_paths[a_key])
            else:
                if len(self.input_paths[a_key]) != first_length:
                    raise ValueError(f"the length of input_dir type {a_key} does not match the length of the first key!"
                                     f" All input keys must have the same length.")

        # Cycle over the output options, making paths where necessary
        for a_key in output_dirs.keys():
            # If it's just one file then we wanna keep it around
            if output_dirs[a_key].is_file():
                self.output_paths = output_dirs[a_key]

            # Otherwise, for dirs, we check that it exists, make it if not, and save it
            else:
                output_dirs[a_key].mkdir(parents=True, exist_ok=True)
                self.output_paths = output_dirs[a_key]

    @staticmethod
    def _check_keys(required, input_dict):
        """Checks that an input dictionary has the correct keys and the correct number of keys."""
        if required is not None:
            if input_dict is None:
                raise ValueError(f"class requires that paths {required} are specified, but none were!")

            if len(input_dict) != len(required) or not all(k in required for k in input_dict.keys()):
                raise ValueError(f"a dictionary of paths does not have the required number of/names of keys! \n"
                                 f"required keys: {required} \n"
                                 f"actual keys: {input_dict.keys()}")

    @staticmethod
    def _open(path: Path):
        """Cheeky function to read in Gaia data from a pickle file or a JSON."""
        mode = path.suffix

        if mode == 'pickle':
            with open(str(path.resolve()), 'rb') as handle:
                return pickle.load(handle)
        elif mode == 'json':
            return pd.read_json(path)
        else:
            raise ValueError("Specified file type not supported! Must be either pickle or json.")


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
                 calculate_cluster_stats: bool = True,):
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
            calculate_cluster_stats (bool): whether or not to calculate cluster statistics.
                Default: True

        """

        super().__init__(names, input_dirs, input_patterns=input_patterns, output_dirs=output_dirs, verbose=verbose,
                         check_input_shape=True,
                         required_input_keys=required_input_keys,
                         required_output_keys=required_output_keys)

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
        self._save_data_gaia_views = 'data' in required_output_keys
        self._n_extra_info = len(extra_returned_info)  # Whether the algorithm returns anything extra
        self._extra_info_names = extra_returned_info

        # We save a cluster array if there's required output key info or extra info incoming.
        # The cluster dataframe is individual information on each cluster, which can include:
        # - Temporary IDs
        # - Calculated statistics
        # - Extra information from the algorithm (e.g. means/stds from GMMs)
        self._save_cluster_info = 'cluster' in required_output_keys
        self._calculate_cluster_stats = calculate_cluster_stats

    def _get_clusterer_args(self, path: Path, input_number):
        """Cheeky hack that lets me use an extra input argument if necessary on certain functions (like GMMs)."""
        main_data = self._open(path)

        if self.third_input_name is None:
            return [main_data]
        else:
            third_data = self._open(self.input_paths[self.third_input_name][input_number])
            return [main_data, third_data]

    def apply(self):
        """Applies an algorithm to the pre-specified clusters."""
        completed_steps = 0
        total_steps = len(self.input_paths['data']) * self.n_kwarg_sets

        iteration_start = datetime.datetime.now()

        # Loop over all the different fields
        for input_number, a_field_name, in enumerate(self.names):

            a_data_gaia_path = self.input_paths['data'][input_number]
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
                self._save_clustering_result(data_gaia, algorithm_return, a_field_name, i_parameter_set)
                self._save_clustering_time(a_field_name, i_parameter_set, runtime)

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

        """
        name = f"{field_name}_{run_name}"

        # De-compose the returned list
        labels = algorithm_return[0]
        probabilities = algorithm_return[1]

        # Save the labels and probabilities arrays
        with open(self.output_paths['labels'] / Path(f"{name}_labels.json"), 'w') as json_file:
            json.dump(labels, json_file)

        if self._save_probabilities:
            with open(self.output_paths['probabilities'] / Path(f"{name}_probs.json"), 'w') as json_file:
                json.dump(labels, json_file)

        # Also save either mini views of data_gaia or cluster info
        if self._save_data_gaia_views or self._save_cluster_info:

            # Grab stuff we need
            all_cluster_indices = np.unique(labels)
            all_cluster_indices = all_cluster_indices[all_cluster_indices != -1]

            parameter_frames_to_concatenate = []

            # Run over all the clusters
            for a_label in all_cluster_indices:
                # Name the cluster!!!!
                cluster_id = name + f"_{a_label}"

                # Calculate all statistics with ocelot
                a_dict = {'field': field_name,
                          'run': run_name,
                          'cluster_label': a_label}

                # Make a baby data gaia if it's needed
                if self._save_data_gaia_views or self._calculate_cluster_stats:
                    baby_data_gaia = data_gaia.loc[labels == a_label].reset_index(drop=True)

                    # Calculate cluster stats
                    if self._calculate_cluster_stats:
                        a_dict.update(ocelot.calculate.all_statistics(
                            baby_data_gaia, membership_probabilities=probabilities))

                    # Save the small data_gaia view
                    if self._save_data_gaia_views:
                        baby_data_gaia.to_json(self.output_paths['data'] / Path(f"{cluster_id}_data.json"))

                    del baby_data_gaia
                    gc.collect()

                parameter_frames_to_concatenate.append(pd.DataFrame(a_dict, index=[0]))

            # Concatenate into a DataFrame
            final_cluster_info = pd.concat(parameter_frames_to_concatenate, ignore_index=True)

            # Add extra info if desired
            if self._n_extra_info > 0:

                extra_columns_for_cluster_info = {}

                for index, an_extra_column in self._extra_info_names:
                    extra_columns_for_cluster_info[an_extra_column] = algorithm_return[index]

                extra_columns_for_cluster_info = pd.DataFrame(extra_columns_for_cluster_info)
                final_cluster_info = final_cluster_info.join(extra_columns_for_cluster_info)

            # Concatenate into a dataframe!
            if self._save_cluster_info:
                final_cluster_info.to_csv(self.output_paths['clusters'] / Path(f"{name}_clusters.csv"))

    def _save_clustering_time(self,
                              field_name: Union[str, int],
                              run_name: Union[str, int],
                              time: float, ):
        """Saves the timing results of a clustering analysis.

        Args:
            names (list-like): names of the fields to save the result for.
            times (list-like): times of the fields to save the result for.

        """
        # Don't over-write (just append) if the file already exists
        if self.output_paths['times'].exists():
            kwargs = {'mode': 'a', 'header': False}
        else:
            kwargs = {}

        pd.DataFrame({'field': field_name, 'run': run_name, 'time': time}).to_csv(self.output_paths['times'], **kwargs)


default_rescale_kwargs = {
    'columns_to_rescale': ('lon', 'lat', 'pmlon', 'pmlat', 'parallax'),
    'column_weights': (1., 1., 1., 1., 1.),
    'scaling_type': 'robust',
    'concatenate': True,
    'return_scaler': True,
}


class PreprocessingPipeline(Pipeline):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 cuts: dict = None,
                 centers: Union[list, tuple, np.ndarray] = None,
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

        self.centers = centers

        self.rescale_kwargs = default_rescale_kwargs
        self.rescale_kwargs.update(kwargs_for_rescaler)

        if self.verbose:
            print("Preprocessing pipeline initialised!")

    def apply(self):
        """Applies the pre-processor."""
        # Cycle over each cluster, applying all of the required pre-processing steps
        for a_path, a_center, a_name in zip(self.input_paths['data'], self.centers, self.names):

            if self.verbose:
                print(f"Working on field {a_name}...")
                print(f"  cutting dataset")

            data_gaia = self._open(a_path)

            # Apply cuts and scaling
            data_gaia = ocelot.cluster.cut_dataset(data_gaia, self.cuts)

            if self.verbose:
                print(f"  re-centering dataset")
            data_gaia = ocelot.cluster.recenter_dataset(data_gaia, center=a_center)

            if self.verbose:
                print(f"  re-scaling dataset")
            data_rescaled, scaler = ocelot.cluster.rescale_dataset(data_gaia, **self.rescale_kwargs)

            if self.verbose:
                print(f"  saving them both!")

            data_gaia.to_json(self.output_paths['cut'] / Path(f"{a_name}_cut.json"))

            with open(self.output_paths['scaler'] / Path(f"{a_name}_scaler.pickle"), 'wb') as handle:
                pickle.dump(scaler, handle, pickle.HIGHEST_PROTOCOL)

            with open(self.output_paths['rescaled'] / Path(f"{a_name}_rescaled.json"), 'w') as json_file:
                json.dump(data_rescaled, json_file)

            del data_gaia, scaler, data_rescaled
            gc.collect()

            if self.verbose:
                print(f"  both saved successfully!")

        if self.verbose:
            print(f"  pre-processing is complete!")
