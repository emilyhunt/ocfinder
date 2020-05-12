"""Functions for running and using UPMASK."""

import ocelot
import numpy as np
import pandas as pd
import time

from .pipeline import ClusteringAlgorithm
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

# R stuff for UPMASK
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
upmask = importr('UPMASK')


default_upmask_kwargs = {
    'positionDataIndexes': ro.IntVector((1, 2)),
    'photometricDataIndexes': ro.IntVector((3, 5, 7)),
    'photometricErrorDataIndexes': ro.IntVector((4, 6, 8)),
    'nRuns': 1,
    'maxIter': 20,
    'fileWithHeader': True,
    'dimRed': 'None',
    'nDimsToKeep': 3,
    'verbose': False,
    'runInParallel': True,
}

default_result_plot_kwargs = {
    'clip_to_fit_clusters': True,
    'cmd_plot_y_limits': [8, 18],
    'figure_title': "UPMASK results for an open cluster",
    'show_figure': True,
    'figure_size': (10, 10),
}


def _get_upmask_cuts(data_gaia, labels, cluster_label, probabilities=None, mode='tcg+18_plus_pmra', debug=False):
    """Returns a dict to pass to run_upmask containing parameter cuts compatible with ocelot.cluster.cut_dataset.
    """
    # First, let's use ocelot to calculate stats for our cluster
    cluster_stars = labels == cluster_label
    data_cluster = data_gaia.loc[cluster_stars].reset_index(drop=True)

    if probabilities is not None:
        probabilities = probabilities[cluster_stars]

    stats_cluster = ocelot.calculate.all_statistics(data_cluster, membership_probabilities=probabilities)

    if debug:
        for a_key in stats_cluster.keys():
            print(f"{a_key}: {stats_cluster[a_key]}")

    # Now, let's define some cuts based on the input_mode we want to use
    # Cuts like Cantat-Gaudin et al. 2018 (DR2 view of OCs)
    if mode == 'tcg+18':
        cuts = {
            'ra': stats_cluster['ra'] + stats_cluster['ang_radius_t'] * np.asarray([-2, 2]),
            'dec': stats_cluster['dec'] + stats_cluster['ang_radius_t'] * np.asarray([-2, 2]),
            'phot_g_mean_mag': [-np.inf, 18],
            'parallax': stats_cluster['parallax'] + np.asarray([-0.5, 0.5]),
        }

    # Cuts like Cantat-Gaudin but with proper motion a little restricted too
    elif mode == 'tcg+18_plus_pmra':
        cuts = {
            'ra': stats_cluster['ra'] + stats_cluster['ang_radius_t'] * np.asarray([-3, 3]),
            'dec': stats_cluster['dec'] + stats_cluster['ang_radius_t'] * np.asarray([-3, 3]),
            'phot_g_mean_mag': [-np.inf, 18],
            'parallax': stats_cluster['parallax'] + np.asarray([-0.5, 0.5]),
            'pmra': stats_cluster['pmra'] + np.asarray([-5, 5]),
            'pmdec': stats_cluster['pmdec'] + np.asarray([-5, 5]),
        }

    # Like Cantat-Gaudin but changed to work specifically with CAC20 results (so, uses r_50)
    elif mode == 'cac20_restricted':
        cuts = {
            'ra': stats_cluster['ra'] + stats_cluster['ang_radius_50'] * np.asarray([-6, 6]),
            'dec': stats_cluster['dec'] + stats_cluster['ang_radius_50'] * np.asarray([-6, 6]),
            'phot_g_mean_mag': [-np.inf, 18],
            'parallax': stats_cluster['parallax'] + np.asarray([-0.5, 0.5]),
            'pmra': stats_cluster['pmra'] + np.asarray([-5, 5]),
            'pmdec': stats_cluster['pmdec'] + np.asarray([-5, 5]),
        }
    else:
        raise ValueError("selected input_mode is not supported!")

    return cuts


def _read_multiple_results(filenames, n_stars, cluster_index_to_assign):
    """Combines multiple UPMASK result CSVs into one label array.

    Notes:
        This function IS NOT VALID when multiple clusters are in one field! Instead, it would need some way to keep
        track of which cluster class is being referred to and where. At the moment, it assumes that all cluster
        probabilities contribute towards the same (one) cluster.

    """
    # Make some blank arrays
    probabilities = np.zeros((len(filenames), n_stars))
    labels = np.zeros((len(filenames), n_stars), dtype=int)

    # Read in every file and keep the two things we actually care about from UPMASK output
    for i, a_file in enumerate(filenames):
        a_result = pd.read_csv(a_file, sep='\t')
        probabilities[i, :] = a_result['probability']
        labels[i, :] = a_result['class']
        del a_result

    # Turn this into mean values
    probabilities_reduced = np.mean(probabilities, axis=0)

    # Grab every star that has a label that field
    labels_reduced = np.where(np.any(labels != 0, axis=0), cluster_index_to_assign, -1)

    # Turn this into mean values and return
    return probabilities_reduced, labels_reduced


def _find_cluster(data_gaia, cuts, upmask_kwargs, n_iterations: int = 5, cluster_index_to_assign=0):
    """Run UPMASK on a Gaia field and returns UPMASK clustering probabilities."""
    # Initialise empty columns
    data_gaia['probabilities'] = 0.
    data_gaia['labels'] = 0

    # Apply cuts to data_gaia
    data_gaia_cut, data_gaia_dropped = ocelot.cluster.cut_dataset(data_gaia, parameter_cuts=cuts,
                                                                  return_cut_stars=True, reset_index=False)

    # If this leaves us without any stars, then the input is bad and we should return nada anyway
    # Deal with if there are no clusters passed here
    if len(data_gaia_cut) < 1:
        return -1 * np.ones(data_gaia.shape[0]), np.zeros(data_gaia.shape[0])

    # Turn the cut data_gaia into a temporary file
    data_gaia_small = \
        data_gaia_cut[['ra', 'dec', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'parallax', 'parallax_error']]

    # Grab a covariance matrix, albeit only if we have covariance information
    if 'pmra_pmdec_corr' in data_gaia_cut.keys():
        covariance_matrix = ocelot.cluster.generate_gaia_covariance_matrix(data_gaia_cut)
    else:
        covariance_matrix = np.zeros((3, 3), dtype=int)
        np.fill_diagonal(covariance_matrix, 1)
        covariance_matrix = np.tile(covariance_matrix, (len(data_gaia_cut), 1, 1))

    # And lastly, keep a list of filenames around
    filenames = [f'temp_out_{i}.csv' for i in range(n_iterations)]

    # Time to run upmask!!!
    for i in range(n_iterations):
        # Re-sample the parallax, pmra and pmdec errors in data_gaia_small
        with pd.option_context("mode.chained_assignment", None):
            data_gaia_small[['pmra', 'pmdec', 'parallax']] = \
                ocelot.cluster.resample_gaia_astrometry(data_gaia_cut, covariance_matrix)

        data_gaia_small.to_csv('./temp.csv', sep=' ')

        # Try to run UPMASK on this file
        upmask.UPMASKfile('temp.csv',
                          filenames[i],
                          **upmask_kwargs)

    # Read in the results and write them to the data frames
    with pd.option_context("mode.chained_assignment", None):
        data_gaia_cut[['probabilities', 'labels']] = np.asarray(
            _read_multiple_results(filenames, data_gaia_cut.shape[0],
                                   cluster_index_to_assign=cluster_index_to_assign)).T
        data_gaia_dropped[['probabilities', 'labels']] = np.asarray(
            [np.zeros(data_gaia_dropped.shape[0]), np.ones(data_gaia_dropped.shape[0]) * -1]).T

    # Re-combine the data frames, sorted back in order again
    data_gaia = pd.concat([data_gaia_dropped, data_gaia_cut]).sort_index()

    # Get probabilities & labels in the right shape
    probabilities, labels = data_gaia[['probabilities', 'labels']].to_numpy().T

    return labels, probabilities


def run_upmask(data_gaia,
               labels,
               labels_to_find,
               probabilities=None,
               cut_mode='cac20_restricted',
               n_iterations: int = 5,
               locally_verbose: bool = True,
               **upmask_kwargs_to_overwrite):
    """Applies UPMASK to validate a cluster candidate. Returns accurate UPMASK labels for clusters in a field.
    In cases when a star has an equal probability of being a member of two or more clusters, then the cluster with the
    lowest cluster label is preferred.

    Note that this could get extremely slow if fed too many clusters, so be careful!
    """
    if locally_verbose:
        print(f"Applying UPMASK to clusters in a field!")

    upmask_kwargs = default_upmask_kwargs
    upmask_kwargs.update(upmask_kwargs_to_overwrite)

    # Deal with if there are no clusters passed here
    if len(labels_to_find) < 1:
        return -1 * np.ones(labels.shape[0]), np.zeros(labels.shape[0])

    # Otherwise, if there are any, we cycle over all the clusters, having a generally quite fun time really
    new_cluster_labels = -1 * np.ones((labels.shape[0], labels_to_find.shape[0]), dtype=int)
    new_cluster_probabilities = np.zeros((labels.shape[0], labels_to_find.shape[0]), dtype=float)

    for i, a_label in enumerate(labels_to_find):

        if locally_verbose:
            print(f"  running UPMASK on cluster {a_label}...")

        start = time.time()

        # Grab the cuts to use
        cuts = _get_upmask_cuts(data_gaia, labels, a_label, probabilities=probabilities, mode=cut_mode,)

        # Run UPMASK
        new_cluster_labels[:, i], new_cluster_probabilities[:, i] = _find_cluster(
            data_gaia, cuts, upmask_kwargs, n_iterations=n_iterations,
            cluster_index_to_assign=a_label)

        if locally_verbose:
            print(f"  finished applying UPMASK to cluster in {(time.time() - start)/60:.2f} minutes.")

    # Find the best cluster for every star
    best_cluster = np.argmax(new_cluster_probabilities, axis=1)

    # Hence, we can make a new label and probability arrays
    indexer = np.arange(labels.shape[0])

    return new_cluster_labels[indexer, best_cluster], new_cluster_probabilities[indexer, best_cluster]


class UPMASKPipeline(ClusteringAlgorithm):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, list]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 user_kwargs: Optional[Union[List[dict], Tuple[dict]]] = None,
                 max_cluster_size_to_save: int = 10000,
                 valid_cluster_key: str = 'valid_total'):
        """Your best friend if you want to test clustering results with a basic first principles cluster detector.

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
        super().__init__(run_upmask,
                         user_kwargs,
                         names,
                         input_dirs,
                         input_patterns=input_patterns,
                         output_dirs=output_dirs,
                         verbose=verbose,
                         required_input_keys=['cut', 'labels', 'cluster_list'],
                         required_output_keys=['labels', 'probabilities',
                                               'cluster_data', 'cluster_list', ],
                         max_cluster_size_to_save=max_cluster_size_to_save,
                         input_shapes_to_not_check=['cut'])

        # Overwrite a couple of superclass things because GMMs are fucking weird - they have lots of extra inputs that
        # I want to make super sure are in the correct order
        self.n_separate_runs = len(self.algorithm_kwargs)
        self.valid_cluster_key = valid_cluster_key
        self.extra_input_names = ['labels', 'cluster_list']
        self.input_paths['rescaled'] = self.input_paths['cut']  # Future me is gonna hate this bullshit trick lol

        # Quick manual check that there are enough cut dataframes to go around as this isn't checked above
        labels_to_cut_ratio = len(self.input_paths['labels']) / len(self.input_paths['cut'])
        if labels_to_cut_ratio != self.n_separate_runs:
            raise ValueError(f"mismatch between the number of 'labels'/'cluster_list' inputs and the number of 'cut' "
                             f"inputs! Given that you have {self.n_separate_runs} to process, there should be "
                             f"{self.n_separate_runs}x more labels/cluster_list entries than there are cut_data "
                             f"entries, but there are actually only {labels_to_cut_ratio} as many.")

    def _get_clusterer_args(self, path: Path, input_number, parameter_number, initial_run_number):
        """Re-defined here because UPMASK is really weird!"""
        # Read in the DataFrame
        if path.suffix == '.feather' or path.suffix == '.csv':
            main_data = self._open(path)
        else:
            raise ValueError("UPMASK requires a DataFrame in data_gaia style, but the input you've passed doesn't "
                             "appear to be a table (i.e. it's not .feather or .csv.)")

        if 'pmra_pmdec_corr' not in main_data.keys():
            print("  WARNING: no covariances found!\n    Falling back on uncovariant random draws of data.\n"
                  "    This is BAD, because Gaia data axes are highly covariant!\n    Be concerned!")

        to_return = [main_data]

        # Next, let's deal with the labels
        index_to_read_in = self.n_separate_runs * input_number + parameter_number - initial_run_number
        to_return.append(self._open(self.input_paths['labels'][index_to_read_in]).values)

        # Finally, we need the cluster_list's valid labels
        cluster_list = self._open(self.input_paths['cluster_list'][index_to_read_in])
        to_return.append(
            cluster_list.loc[cluster_list[self.valid_cluster_key], 'cluster_label'].to_numpy())

        return to_return
