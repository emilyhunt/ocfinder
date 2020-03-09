"""Functions to support the use of DBSCAN to find open clusters."""

import ocelot
import numpy as np
import pandas as pd
import json
import gc
import datetime

from pathlib import Path
from typing import Union, List, Tuple
from scipy import sparse
from sklearn.cluster import DBSCAN

from .utilities import print_itertime


def dbscan_preprocessing(input_dir: Path,
                         output_dir: Path,
                         epsilon_file: Path,
                         allow_epsilon_file_overwrite: bool = False,
                         files_to_run_on: Union[int, str] = 'all',
                         min_samples: int = 10,
                         acg_repeats: Union[List[int], Tuple[int], np.ndarray] = (10,),
                         verbose: bool = True) -> None:
    """Creates pre-processed sparse matrices and distance arrays, and calculates epsilon estimates.

    Args:
        input_dir (pathlib.Path): directory to find rescaled arrays in (must have filenames ending in _rescaled.json)
        output_dir (pathlib.Path): directory to place all output of this function in.
        epsilon_file (pathlib.Path): location of the epsilon csv file to write to.
        allow_epsilon_file_overwrite (bool): whether or not to allow overwrites of the epsilon .csv file.
            Default: False (will raise an error if attempted)
        files_to_run_on (str or int): if 'all', run on everything. Otherwise, if int, run on everything upto that file
            number.
            Default: 'all'
        min_samples (int): min_samples parameter to calculate epsilon values for.
            Default: 10
        acg_repeats (list-like): an array or list where each element is an int specifying how many ACG epsilon repeats
            to do.
            Default: (10,) (i.e. just do one epsilon calculation for 10 repeats)
        verbose (bool): whether or not to print updates and an ETA for finishing.
            Default: True (recommended)

    Returns:
        None, but sparse matrices, distance arrays and field model epsilon diagnostic plots will be written to
            output_dir, and an epsilon .csv file of epsilon estimates made (appended to if overwrites allowed) will be
            made at epsilon_file.

    """
    # Grab all of the files to pre-process, including cutting the list of files to run on if requested
    files_to_preprocess = list(input_dir.glob("*_rescaled.json"))

    if files_to_run_on != 'all':
        files_to_preprocess = files_to_preprocess[:files_to_run_on]

    # Also estimate how long the file extensions are so that we can extract names correctly
    extension_length = len("_rescaled.json")

    # Initialise the epsilon file if it doesn't already exist
    if epsilon_file.exists():
        if not allow_epsilon_file_overwrite:
            raise ValueError("specified location of epsilon estimates for the field (aka epsilon_file) already exists! "
                             "Set allow_epsilon_file_overwrite=True to disable this warning message and allow "
                             "the file to be appended to.")
        new_file = False
    else:
        new_file = True

    # Declare some useful stuff before we begin, including names of field model parameters
    epsilon_names = ['eps_c', 'eps_n1', 'eps_n2', 'eps_n3', 'eps_f']
    parameter_names = ['field_constant', 'field_dimension', 'cluster_constant', 'cluster_dimension', 'cluster_fraction']

    iteration_start = datetime.datetime.now()
    completed_steps = 0
    total_steps = len(files_to_preprocess)

    # Calculate epsilon with all methods
    for a_file in files_to_preprocess:

        name = str(a_file).rsplit("/")[-1][:-extension_length]

        if verbose:
            print(f"Working on field {name}...")
            print(f"  reading in file")

        # First off, let's make us a lovely sparse matrix and distance array
        with open(str(a_file.resolve()), 'r') as json_file:
            data_rescaled = json.load(json_file)

        if verbose:
            print(f"  calculating sparse matrix and nn distance array")
        sparse_matrix, distances = ocelot.cluster.precalculate_nn_distances(
            data_rescaled, n_neighbors=min_samples, return_sparse_matrix=True, return_knn_distance_array=True)

        # Now, let's calculate the desired epsilon values
        a_epsilon_dict = {'id': name}

        # First, for the acg method
        for a_acg_repeats in acg_repeats:
            if verbose:
                print(f"  calculating epsilon acg for {a_acg_repeats} random draws")

            start = datetime.datetime.now()
            a_epsilon_dict[f"acg_{a_acg_repeats}"] = ocelot.cluster.epsilon.acg18(
                data_rescaled, distances, n_repeats=a_acg_repeats, min_samples=min_samples)
            a_epsilon_dict[f"time_acg_{a_acg_repeats}"] = (datetime.datetime.now() - start).total_seconds()

        # Now, for my method
        if verbose:
            print(f"  calculating field model epsilon values")

        # Do it once just to time it precisely (lol)
        start = datetime.datetime.now()
        result, epsilons, parameters, n_members = ocelot.cluster.epsilon.field_model(
            distances, min_samples=min_samples)

        a_epsilon_dict.update(field_model_success=result,
                              **dict(zip(epsilon_names, epsilons)),
                              **dict(zip(parameter_names, parameters)),
                              estimated_n_members=n_members)

        a_epsilon_dict[f"time_field_model"] = (datetime.datetime.now() - start).total_seconds()

        # Then do the whole thing again just to plot it (so fucking lol)
        if verbose:
            print(f"  plotting the output")

        ocelot.cluster.epsilon.field_model(
            distances,
            min_samples=min_samples,
            make_diagnostic_plot=True,
            figure_title=f"nearest neighbour distances for field {name}",
            number_of_derivatives=2,
            save_name=output_dir / Path(f"{name}_nn_distances.png")
        )

        # Save the data
        if verbose:
            print(f"  saving the output incrementally")

        # Distance info & sparse matrix
        with open(output_dir / Path(f"{name}_distances.json"), 'w') as json_file:
            json.dump(distances, json_file)

        sparse.save_npz(output_dir / Path(f"{name}_matrix.npz"), sparse_matrix)

        # Epsilon values CSV (making it freshly if needed)
        if new_file:
            epsilon_df = pd.DataFrame(pd.Series(a_epsilon_dict))
            new_file = False
        else:
            epsilon_df = pd.read_csv(epsilon_file)
            epsilon_df = epsilon_df.append(pd.Series(a_epsilon_dict), ignore_index=True)
        epsilon_df.to_csv(epsilon_file)

        # Paranoid garbage collection
        del distances, sparse_matrix, epsilon_df, data_rescaled
        gc.collect()

        # Output info
        completed_steps += 1
        if verbose:
            print_itertime(iteration_start, completed_steps, total_steps)

    print("All DBSCAN preprocessing is completed!")


default_dbscan_kwargs = {
    'n_jobs': -1,
    'min_samples': 10,
}


def run_dbscan(data: Union[sparse.csr_matrix, np.ndarray], epsilon_value: float, **kwargs_for_algorithm):
    """Runs DBSCAN on a field, given an arbitrary number of epsilon values to try.

    Args:
        data (scipy.sparse.csr_matrix or np.ndarray): data to use. If csr_matrix, then it is presumed to be a
            pre-computed sparse matrix and metric=precomputed will be used. Otherwise if np.ndarray, assumed to be an
            array of shape (n_samples, n_features), and nearest neighbor analysis will be performed manually.
        epsilon_value (float): the value for epsilon to run with.
        **kwargs_for_algorithm: additional kwargs to pass to sklearn.cluster.DBSCAN.

    Returns:
        label array, and None since DBSCAN does not generate probabilities!

    """
    dbscan_kwargs = default_dbscan_kwargs
    dbscan_kwargs.update(kwargs_for_algorithm)

    # Decide on whether the clusterer will be ran with
    if type(data) == np.ndarray:
        clusterer = DBSCAN(metric='euclidean', eps=epsilon_value, **dbscan_kwargs)
    else:
        clusterer = DBSCAN(metric='precomputed', eps=epsilon_value, **dbscan_kwargs)

    return clusterer.fit_predict(data), None
