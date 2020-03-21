"""Functions to support the use of DBSCAN to find open clusters."""

import ocelot
import numpy as np
import pandas as pd
import json
import gc
import datetime
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict
from scipy import sparse
from sklearn.cluster import DBSCAN

from .utilities import print_itertime
from .pipeline import Pipeline, ClusteringAlgorithm


class DBSCANPreprocessor(Pipeline):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 min_samples: int = 10,
                 acg_repeats: Union[list, tuple, np.ndarray] = (10,)):
        """Pre-processor for taking Gaia data as input and calculating all required stuff.

        Args:
            names (list-like): list of names of fields to run on to save.
            input_dirs (dict of paths or lists of paths): input directories. You have three options:
                - If a path to one file, just one path will be saved
                - If a list of paths, this list will be saved
                - If a path to a directory, then files with a suffix specified by input_patterns[the_dict_key] will be
                  saved.
                Must specify 'rescaled'.
            input_patterns (dict of str): for input_dirs that are a path to a directory, this is the pattern to look
                for. Can just be "*" to look for all files.
                Default: None
            output_dirs (dict of paths): output directories to write to. Must specify 'epsilon', 'plots', 'sparse'.
            verbose (bool): whether or not to run in verbose mode with informative print statements.
                Default: True
            min_samples (int): min_samples parameter to calculate epsilon values for.
                Default: 10
            acg_repeats (list-like): an array or list where each element is an int specifying how many ACG epsilon
                repeats to do.
                Default: (10,) (i.e. just do one epsilon calculation for 10 repeats)


        """
        super().__init__(names,
                         input_dirs,
                         input_patterns=input_patterns, output_dirs=output_dirs, verbose=verbose,
                         required_input_keys=['rescaled'],
                         required_output_keys=['epsilon', 'plots', 'sparse'],
                         check_input_shape=True)

        self.min_samples = min_samples
        self.acg_repeats = acg_repeats

        # Some useful things for the field model
        self.epsilon_names = ['eps_c', 'eps_n1', 'eps_n2', 'eps_n3', 'eps_f']
        self.parameter_names = ['field_constant', 'field_dimension', 'cluster_constant', 'cluster_dimension',
                                'cluster_fraction']

        if self.verbose:
            print("DBSCAN preprocessing pipeline initialised!")

    def _save_epsilon_dataframe(self, data_to_save: dict, field_name: str):
        """Incrementally saves values to the dataframe of epsilon estimates. Must always have the same columns!"""
        # Work out if it's a new file or not
        epsilon_filename = self.output_paths['epsilon'] / Path(f'{field_name}_epsilon.csv')

        # And save!
        pd.DataFrame(data_to_save, index=[0]).to_csv(epsilon_filename, index=False)

    def apply(self, start: int = 0):
        """Applies the pre-processor.

        Args:
            start (int): the cluster number in self.names to start with.
                Default: 0

        """
        completed_steps = start
        total_steps = len(self.names)
        iteration_start = datetime.datetime.now()

        # Cycle over each cluster, applying all of the required pre-processing steps
        for a_path, a_name in zip(self.input_paths['rescaled'][start:], self.names[start:]):

            if self.verbose:
                print(f"-- {datetime.datetime.today()}")
                print(f"Working on field {a_name}...")
                print("  opening the field's data")

            # Open up the data
            data_rescaled = self._open(a_path).values

            # Make a nearest neighbor distances array upto min_samples
            if self.verbose:
                print("  pre-calculating nearest neighbor distances for the fields")
            start = datetime.datetime.now()
            sparse_matrix, nn_distances = ocelot.cluster.precalculate_nn_distances(
                data_rescaled, self.min_samples, return_sparse_matrix=True, return_knn_distance_array=True)
            runtime_nn_distance = (datetime.datetime.now() - start).total_seconds()

            # Calculate the ACG epsilon
            if self.verbose:
                print("  calculating epsilons for the acg method")
            start = datetime.datetime.now()
            a_epsilon_dict = ocelot.cluster.epsilon.acg18(
                data_rescaled, nn_distances, n_repeats=self.acg_repeats, min_samples=self.min_samples,
                return_std_deviation=True).to_dict()
            a_epsilon_dict['runtime_acg'] = (datetime.datetime.now() - start).total_seconds()

            del data_rescaled
            gc.collect()

            # Calculate the field model epsilon
            if self.verbose:
                print("  calculating epsilons for the field model method")
            start = datetime.datetime.now()
            result, epsilons, parameters, n_members = ocelot.cluster.epsilon.field_model(
                nn_distances, min_samples=self.min_samples)

            a_epsilon_dict.update(field_model_success=result,
                                  **dict(zip(self.epsilon_names, epsilons)),
                                  **dict(zip(self.parameter_names, parameters)),
                                  estimated_n_members=n_members)

            a_epsilon_dict['runtime_field_model'] = (datetime.datetime.now() - start).total_seconds()
            a_epsilon_dict['runtime_nn_distance'] = runtime_nn_distance

            # Now, let's make a diagnostic plot of the results (we do the whole thing again just to plot it because
            # my old API fucking sucked lmao)
            if self.verbose:
                print(f"  plotting the output")

            ocelot.cluster.epsilon.field_model(
                nn_distances,
                min_samples=self.min_samples,
                make_diagnostic_plot=True,
                figure_title=f"nearest neighbour distances for field {a_name}",
                number_of_derivatives=2,
                save_name=self.output_paths['plots'] / Path(f"{a_name}_nn_distances.png"),
                show_figure=False
            )

            # Save the data
            if self.verbose:
                print(f"  saving the output incrementally")

            self._save_epsilon_dataframe(a_epsilon_dict, a_name)

            sparse.save_npz(self.output_paths['sparse'] / Path(f"{a_name}_matrix.npz"), sparse_matrix)

            # Memory management
            del nn_distances, sparse_matrix
            plt.close('all')
            gc.collect()

            # Output
            completed_steps += 1
            if self.verbose:
                print_itertime(iteration_start, completed_steps, total_steps)

        if self.verbose:
            print(f"  DBSCAN pre-processing is complete!")


default_dbscan_kwargs = {
    'n_jobs': -1,
    'min_samples': 10,
}


def run_dbscan(data: Union[sparse.csr_matrix, np.ndarray], data_epsilon: pd.DataFrame,
               epsilon_value: str = 'acg_5', **kwargs_for_algorithm):
    """Runs DBSCAN on a field, given an arbitrary number of epsilon values to try.

    Args:
        data (scipy.sparse.csr_matrix or np.ndarray): data to use. If csr_matrix, then it is presumed to be a
            pre-computed sparse matrix and metric=precomputed will be used. Otherwise if np.ndarray, assumed to be an
            array of shape (n_samples, n_features), and nearest neighbor analysis will be performed manually.
        data_epsilon (pd.DataFrame): a DataFrame of shape (doesn't matter lol, 1) containing epsilon estimates to use.
        epsilon_value (str): the key into data_epsilon to use as our epsilon value.
        **kwargs_for_algorithm: additional kwargs to pass to sklearn.cluster.DBSCAN.

    Returns:
        label array, and None since DBSCAN does not generate probabilities!

    """
    dbscan_kwargs = default_dbscan_kwargs
    dbscan_kwargs.update(kwargs_for_algorithm)
    dbscan_kwargs.update(eps=float(data_epsilon.loc[0, epsilon_value]))

    # Decide on whether the clusterer will be ran with
    if type(data) == np.ndarray:
        clusterer = DBSCAN(metric='euclidean', **dbscan_kwargs)
    else:
        clusterer = DBSCAN(metric='precomputed', **dbscan_kwargs)

    return clusterer.fit_predict(data), None


class DBSCANPipeline(ClusteringAlgorithm):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 user_kwargs: Optional[Union[List[dict], Tuple[dict]]] = None,
                 max_cluster_size_to_save: int = 10000):
        """Looking to run DBSCAN on a large selection of fields? Then look no further! This pipeline has you covered.

        Args:
            names (list-like): list of names of fields to run on to save.
            input_dirs (dict of paths or lists of paths): input directories. You have three options:
                - If a path to one file, just one path will be saved
                - If a list of paths, this list will be saved
                - If a path to a directory, then files with a suffix specified by input_patterns[the_dict_key] will be
                  saved.
                All key locations must end up having the same shape if check_input_shape=True, which is specified by the
                requirements of the subclass.
                Must specify 'cut', 'rescaled', 'epsilon'. Additionally, 'rescaled' may in fact be a sparse distances
                matrix.
            input_patterns (dict of str): for input_dirs that are a path to a directory, this is the pattern to look
                for. Can just be "*" to look for all files.
                Default: None
            output_dirs (dict of paths): output directories to write to.
            verbose (bool): whether or not to run in verbose mode with informative print statements.
                Default: True
            user_kwargs (list or tuple of dicts, optional): parameter sets to run with. len(user_kwargs) is the number
                of runs that will be performed.
                Default: None (only runs with default parameters once)
            max_cluster_size_to_save (int): maximum cluster size to save baby_data_gaia views for. If 0 or -ve, then we
                also don't calculate any statistics for the cluster, and just return the labels (+probabilities).

        User methods:
            apply(): applies the algorithm to the data specified by input_dirs.

        """

        super().__init__(run_dbscan,
                         user_kwargs,
                         names,
                         input_dirs,
                         input_patterns=input_patterns,
                         output_dirs=output_dirs,
                         verbose=verbose,
                         required_input_keys=['cut', 'rescaled', 'epsilon'],
                         required_output_keys=['labels', 'cluster_data', 'cluster_list', 'times'],
                         max_cluster_size_to_save=max_cluster_size_to_save, )
