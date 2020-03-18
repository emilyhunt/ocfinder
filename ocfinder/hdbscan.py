"""Functions for running HDBSCAN."""

import numpy as np
import hdbscan

from typing import Union, Optional, Dict, List, Tuple
from scipy import sparse
from pathlib import Path
from .pipeline import ClusteringAlgorithm


default_hdbscan_kwargs = {
    'memory': 'hdbscan_cache',
    'core_dist_n_jobs': -1,
    'cluster_selection_method': 'leaf',
    'allow_single_cluster': False,
    'prediction_data': True,
    'min_cluster_size': 40,
    'min_samples': 10,
}


def run_hdbscan(data: Union[sparse.csr_matrix, np.ndarray],
                max_clustered_stars_for_validity: int = 10000,
                **kwargs_for_algorithm):
    """Runs HDBSCAN on a field, given an arbitrary number of epsilon values to try.

    Args:
        data (scipy.sparse.csr_matrix or np.ndarray): data to use. If csr_matrix, then it is presumed to be a
            pre-computed sparse matrix and metric=precomputed will be used. Otherwise if np.ndarray, assumed to be an
            array of shape (n_samples, n_features), and nearest neighbor analysis will be performed manually.
        max_clustered_stars_for_validity (int): threshold at which we don't try to compute DBCV validity scores.
            10,000 requires about 0.75GB of RAM, as a square array has to be made! Setting to 0 is equivalent to
            this being off.
            Default: 10000
        **kwargs_for_algorithm: additional kwargs to pass to hdbscan.HDBSCAN.

    Returns:
        labels (n_samples,), probabilities (n_samples,), persistences (n_clusters,) and validities (n_clusters,).

    """
    hdbscan_kwargs = default_hdbscan_kwargs
    hdbscan_kwargs.update(kwargs_for_algorithm)

    # Decide on whether the clusterer will be ran with
    if type(data) == np.ndarray:
        clusterer = hdbscan.HDBSCAN(metric='euclidean', **hdbscan_kwargs)
    else:
        clusterer = hdbscan.HDBSCAN(metric='precomputed', **hdbscan_kwargs)

    labels = clusterer.fit_predict(data)
    probabilities = clusterer.probabilities_
    persistences = clusterer.cluster_persistence_

    # Only calculate the DBCV statistic if requested by the user
    if max_clustered_stars_for_validity != 0:

        value_counts_excluding_noise = (np.unique(labels, return_counts=True)[1])[1:]
        n_members_total = np.sum(value_counts_excluding_noise, dtype=int)

        # And only if it won't eat too much RAM!
        if n_members_total < max_clustered_stars_for_validity:
            validities = hdbscan.validity.validity_index(data, labels, per_cluster_scores=True, )[1]
        else:
            validities = np.full_like(persistences, np.nan)

    else:
        validities = np.full_like(persistences, np.nan)

    return labels, probabilities, persistences, validities


class HDBSCANPipeline(ClusteringAlgorithm):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 user_kwargs: Optional[Union[List[dict], Tuple[dict]]] = None):
        """Looking to run HDBSCAN on a large selection of fields? Then look no further! This pipeline has you covered.

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
            user_kwargs (list or tuple of dicts, optional): parameter sets to run with. len(user_kwargs) is the number
                of runs that will be performed.
                Default: None (only runs with default parameters once)

        User methods:
            apply(): applies the algorithm to the data specified by input_dirs.

        """

        super().__init__(run_hdbscan,
                         user_kwargs,
                         names,
                         input_dirs,
                         input_patterns=input_patterns,
                         output_dirs=output_dirs,
                         verbose=verbose,
                         required_input_keys=['cut', 'rescaled', 'epsilon'],
                         required_output_keys=['labels', 'probabilities',
                                               'cluster_data', 'cluster_list', 'times'],
                         extra_returned_info=['persistences', 'validities'],
                         calculate_cluster_stats=True,)
