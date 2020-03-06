import numpy as np
import hdbscan

from typing import Union, Optional
from scipy import sparse


default_hdbscan_kwargs = {
    'memory': 'hdbscan_cache',
    'core_dist_n_jobs': -1,
    'cluster_selection_method': 'leaf',
    'allow_single_cluster': False,
    'prediction_data': True,
}


def run_hdbscan(data: Union[sparse.csr_matrix, np.ndarray], min_cluster_size: int, min_samples: Optional[int],
                max_clustered_stars_for_validity: int = 10000,
                **kwargs_for_algorithm):
    """Runs HDBSCAN on a field, given an arbitrary number of epsilon values to try.

    Args:
        data (scipy.sparse.csr_matrix or np.ndarray): data to use. If csr_matrix, then it is presumed to be a
            pre-computed sparse matrix and metric=precomputed will be used. Otherwise if np.ndarray, assumed to be an
            array of shape (n_samples, n_features), and nearest neighbor analysis will be performed manually.
        min_cluster_size (int): the minimum cluster size parameter.
        min_samples (int, optional): the minimum samples parameter. If None, will be the same as min_cluster_size.
        **kwargs_for_algorithm: additional kwargs to pass to hdbscan.HDBSCAN.

    Returns:
        todo fstring

    """
    hdbscan_kwargs = default_hdbscan_kwargs
    hdbscan_kwargs.update(kwargs_for_algorithm, min_cluster_size=min_cluster_size, min_samples=min_samples)

    # Decide on whether the clusterer will be ran with
    if type(data) == np.ndarray:
        clusterer = hdbscan.HDBSCAN(metric='euclidean', **hdbscan_kwargs)
    else:
        clusterer = hdbscan.HDBSCAN(metric='precomputed', **hdbscan_kwargs)

    labels = clusterer.fit_predict(data)
    probabilities = clusterer.probabilities_
    persistences = clusterer.cluster_persistence_

    value_counts_excluding_noise = (np.unique(labels, return_counts=True)[1])[1:]
    n_members_total = np.sum(value_counts_excluding_noise, dtype=int)

    if n_members_total < max_clustered_stars_for_validity:
        validities = hdbscan.validity.validity_index(data, labels, per_cluster_scores=True, )[1]
    else:
        validities = np.full_like(persistences, np.nan)


    # todo: end of play on 6/03/20
