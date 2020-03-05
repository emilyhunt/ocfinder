"""Functions to support the use of DBSCAN to find open clusters."""

import ocelot
import numpy as np
import pandas as pd
import json
import gc
import time

from pathlib import Path
from typing import Union


def dbscan_preprocessing(input_dir: Path, output_dir: Path, files_to_run_on='all', min_samples=10,
                         acg_repeats: Union[list, tuple, np.ndarray] = (10), verbose=True):
    """Creates pre-processed sparse matrices and calculates epsilon estimates."""
    # Grab all of the files to pre-process, including cutting the list of files to run on if requested
    files_to_preprocess = list(input_dir.glob("*_rescaled.json"))

    if files_to_run_on != 'all':
        files_to_preprocess = files_to_preprocess[:files_to_run_on]

    extension_length = len("_rescaled.json")

    epsilons = []

    for a_file in files_to_preprocess:

        name = str(a_file).rsplit("/")[-1][:-extension_length]

        if verbose:
            print(f"Working on field {name}...")
            print(f"  reading in file")

        # First off, let's make us a lovely sparse matrix
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

            start = time.time()
            a_epsilon_dict[f"acg_{a_acg_repeats}"] = ocelot.cluster.epsilon.acg18(
                data_rescaled, distances, n_repeats=a_acg_repeats, min_samples=min_samples)
            a_epsilon_dict[f"time_acg_{a_acg_repeats}"] = time.time() - start

        # Now, for my method
        # Do it once just to time it (lol)
        start = time.time()
        result, epsilons, parameters, n_members = ocelot.cluster.epsilon.field_model(
            distances, min_samples=min_samples)  # todo stopping point on 5/03/20!!!


        # Then do the whole thing again just to plot it (so fucking lol)



    pass


def run_dbscan():
    """Runs DBSCAN on a field, given an arbitrary number of epsilon values to try."""
    pass

