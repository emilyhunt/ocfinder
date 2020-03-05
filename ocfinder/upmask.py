"""Functions for running and using UPMASK."""

import ocelot
import numpy as np
import pandas as pd
import time

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

blanco_1_cuts = {'phot_g_mean_mag': [-np.inf, 18],
                 'parallax': [2, np.inf],
                 'pmra': [15, 25],
                 'pmdec': [-2, 7]}


def _get_upmask_cuts(data_gaia, labels, cluster_label, probabilities=None, mode='tcg+18', debug=False):
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
            'ra': stats_cluster['ra'] + stats_cluster['ang_radius_t'] * np.asarray([-2, 2]),
            'dec': stats_cluster['dec'] + stats_cluster['ang_radius_t'] * np.asarray([-2, 2]),
            'phot_g_mean_mag': [-np.inf, 18],
            'parallax': stats_cluster['parallax'] + np.asarray([-0.5, 0.5]),
            'pmra': stats_cluster['pmra'] + np.asarray([-10, 10]),
            'pmdec': stats_cluster['pmdec'] + np.asarray([-10, 10]),
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


def _find_cluster(data_gaia, cuts, n_iterations: int = 5, plot_cuts=False, plot_result=False,
                  upmask_kwargs_to_overwrite=None, result_plot_kwargs_to_overwrite=None, cluster_index_to_assign=0):
    """Run UPMASK on a Gaia field and returns UPMASK clustering probabilities."""
    # Initialise empty columns
    data_gaia['probabilities'] = 0.
    data_gaia['labels'] = 0

    # Apply cuts to data_gaia
    data_gaia_cut, data_gaia_dropped = ocelot.cluster.cut_dataset(data_gaia, parameter_cuts=cuts,
                                                                  return_cut_stars=True, reset_index=False)

    if plot_cuts:
        ocelot.plot.location(data_gaia_cut, figure_title='Blanco 1, after cuts')

    # Turn the cut data_gaia into a temporary file
    data_gaia_small = \
        data_gaia_cut[['ra', 'dec', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'parallax', 'parallax_error']]

    # Grab a covariance matrix
    covariance_matrix = ocelot.cluster.generate_gaia_covariance_matrix(data_gaia_cut)

    # Overwrite any of my default UPMASK parameters
    upmask_kwargs = default_upmask_kwargs
    if upmask_kwargs_to_overwrite is not None:
        for a_kwarg in upmask_kwargs_to_overwrite.keys():
            upmask_kwargs[a_kwarg] = upmask_kwargs_to_overwrite[a_kwarg]

    # And lastly, keep a list of filenames around
    filenames = [f'temp_out_{i}.csv' for i in range(n_iterations)]

    # Time to run upmask!!!
    i = 0
    while i < n_iterations:
        # Re-sample the parallax, pmra and pmdec errors in data_gaia_small
        with pd.option_context("input_mode.chained_assignment", None):
            data_gaia_small[['pmra', 'pmdec', 'parallax']] = \
                ocelot.cluster.resample_gaia_astrometry(data_gaia_cut, covariance_matrix)

        data_gaia_small.to_csv('./temp.csv', sep=' ')

        # Try to run UPMASK on this file
        upmask.UPMASKfile('temp.csv',
                          filenames[i],
                          **upmask_kwargs)

        i += 1

    # Read in the results and write them to the data frames
    with pd.option_context("input_mode.chained_assignment", None):
        data_gaia_cut[['probabilities', 'labels']] = np.asarray(
            _read_multiple_results(filenames, data_gaia_cut.shape[0],
                                   cluster_index_to_assign=cluster_index_to_assign)).T
        data_gaia_dropped[['probabilities', 'labels']] = np.asarray(
            [np.zeros(data_gaia_dropped.shape[0]), np.ones(data_gaia_dropped.shape[0]) * -1]).T

    # Re-combine the data frames, sorted back in order again
    data_gaia = pd.concat([data_gaia_dropped, data_gaia_cut]).sort_index()

    # Get probabilities & labels in the right shape
    probabilities, labels = data_gaia[['probabilities', 'labels']].to_numpy().T

    # Plot external kwargs if desired
    if plot_result:

        # Overwrite the default kwargs
        result_plot_kwargs = default_result_plot_kwargs
        if result_plot_kwargs_to_overwrite is not None:
            for a_kwarg in result_plot_kwargs_to_overwrite.keys():
                result_plot_kwargs[a_kwarg] = result_plot_kwargs_to_overwrite[a_kwarg]

        fig, ax = ocelot.plot.clustering_result(data_gaia,
                                                labels,
                                                [cluster_index_to_assign],
                                                probabilities,
                                                **result_plot_kwargs)

    return labels, probabilities


def run_upmask(data_gaia,
               labels,
               labels_to_find,
               probabilities=None,
               cut_mode='tcg+18',
               n_iterations: int = 5,
               upmask_kwargs_to_overwrite=None,
               verbose=True):
    """Applies UPMASK to validate a cluster candidate. Returns accurate UPMASK labels for clusters in a field.
    In cases when a star has an equal probability of being a member of two or more clusters, then the cluster with the
    lowest cluster label is preferred.

    Note that this could get extremely slow if fed too many clusters, so be careful!
    """
    if verbose:
        print(f"Applying UPMASK to clusters in a field!")

    # Cycle over all the clusters, having a generally quite fun time really
    new_cluster_labels = np.ones((labels.shape[0], labels_to_find.shape[0]), dtype=int)
    new_cluster_probabilities = np.zeros((labels.shape[0], labels_to_find.shape[0]), dtype=float)

    for i, a_label in enumerate(labels_to_find):

        if verbose:
            print(f"  running UPMASK on cluster {a_label}...")

        start = time.time()

        # Grab the cuts to use
        cuts = _get_upmask_cuts(data_gaia, labels, a_label, probabilities=probabilities, mode=cut_mode,)

        # Run UPMASK
        new_cluster_labels[i], new_cluster_probabilities[i] = _find_cluster(
            data_gaia, cuts, n_iterations=n_iterations, upmask_kwargs_to_overwrite=upmask_kwargs_to_overwrite,
            cluster_index_to_assign=a_label)

        if verbose:
            print(f"  finished applying UPMASK to cluster in {(time.time() - start)/60:.2f} minutes.")

    # Find the best cluster for every star
    best_cluster = np.argmax(new_cluster_probabilities, axis=1)

    # Hence, we can make a new label and probability arrays
    indexer = np.arange(labels.shape[0])

    return new_cluster_labels[indexer, best_cluster], new_cluster_probabilities[indexer, best_cluster]
