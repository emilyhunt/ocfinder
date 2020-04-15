"""Functions for running Gaussian mixture models on a field."""

import ocelot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import gc
import sys
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from typing import Union, Optional, List, Dict, Tuple
from pathlib import Path

from .pipeline import ClusteringAlgorithm, Pipeline, _blank_statistics_dataframe
from .utilities import print_itertime


def tcg_proper_motion_cut(mixture_parameters, cut_parameters):
    """Implements the TCG proper motion dispersion cut.

    Required cut_parameters to use:
        None
    """
    max_pm_dispersion = np.where(mixture_parameters['parallax'] > 0.6703,
                                 5 * np.sqrt(2) / 4.74 * mixture_parameters['parallax'],
                                 1)
    pm_dispersion = np.sqrt(mixture_parameters['pmlon_std'].to_numpy()**2
                            + mixture_parameters['pmlat_std'].to_numpy()**2)
    return pm_dispersion < max_pm_dispersion


def radius_cut(mixture_parameters, cut_parameters):
    """Simple radius cut based on a max radius for a cluster, defined in parsecs.

    Required cut_parameters to use:
        max_radius (pc)
        max_distance (pc) - the maximum distance to calculate for.
    """
    distances = np.clip(1000 / mixture_parameters['parallax'], 0, cut_parameters['max_distance'])
    max_radius = np.arctan(cut_parameters['max_radius'] / distances) * 180 / np.pi

    radius_dispersion = np.sqrt(mixture_parameters['ra_std'].to_numpy()**2
                                + mixture_parameters['dec_std'].to_numpy()**2)

    return radius_dispersion < max_radius


def size_cut(mixture_parameters, cut_parameters):
    """Simple size cut based on a maximum and minimum size for the cluster.

    Required cut_parameters to use:
        min_size
        max_size
    """
    not_too_small = mixture_parameters['n_stars'] > cut_parameters['min_size']
    not_too_big = mixture_parameters['n_stars'] < cut_parameters['max_size']
    return np.asarray(np.logical_and(not_too_big, not_too_small))


def apply_radius_cut(data_gaia, labels, cuts):
    # Todo fstring
    ratios = np.zeros(cuts.shape[0])
    successes = np.zeros(cuts.shape[0], dtype=bool)
    for i, (a_label, a_cut) in enumerate(zip(labels, cuts)):
        matches = labels == a_label
        ra = np.median(data_gaia.loc[matches, 'ra'])
        dec = np.median(data_gaia.loc[matches, 'dec'])
        radii = np.sqrt((data_gaia.loc[matches, 'ra'] - ra) ** 2 + (data_gaia.loc[matches, 'dec'] - dec) ** 2)

        ratios[i] = np.max(radii) / a_cut
        successes[i] = ratios[i] <= 1.

    return ratios, successes


def calculate_cuts(data_gaia, labels, mean, std, counts, cut_parameters):
    # Todo re-factor to not use radius cut (or have a way to turn it off) & fstring

    # Now, let's calculate the cuts we want
    radius_cuts = radius_cut(mean, std, cut_parameters)
    pm_cuts = tcg_proper_motion_cut(mean, std, cut_parameters)

    # And let's calculate dispersions too
    radius_dispersion = np.sqrt(std['ra'] ** 2 + std['dec'] ** 2)
    pm_dispersion = np.sqrt(std['pmra'] ** 2 + std['pmdec'] ** 2)

    # And lastly, we can see which GMMs are compatible with our cuts
    ratios = {}
    good_clusters = {}

    ratios['radius'], good_clusters['radius'] = apply_radius_cut(data_gaia, labels, radius_cuts)
    ratios['pm'] = pm_dispersion / pm_cuts

    good_clusters['pm'] = ratios['pm'] <= 1.
    good_clusters['size'] = np.logical_and(counts >= cut_parameters['min_size'], counts <= cut_parameters['max_size'])
    good_clusters['total'] = np.logical_and(np.logical_and(
        good_clusters['radius'], good_clusters['pm']), good_clusters['size'])

    return good_clusters, ratios


class ValueRemapper:
    def __init__(self, sort_on):
        """Class for re-mapping values to new ones, allowing my bar chart plot to be a hell of a lot prettier. Uses
        list comprehension from:
        https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
        """
        self.from_values = np.arange(sort_on.shape[0])
        self.to_values = np.argsort(sort_on)

    def apply_to_dict(self, *dictionaries):
        """Applies re-mapping to dicts containing multiple arrays, of length n_components."""
        to_return = []

        for a_dict in dictionaries:
            for a_key in a_dict.keys():
                a_dict[a_key] = a_dict[a_key][self.to_values]

            to_return.append(a_dict)

        return to_return

    def apply_to_arrays(self, *arrays):
        """Applies re-mapping to arrays, of length n_components."""
        to_return = []

        for an_array in arrays:
            to_return.append(an_array[self.to_values])

        return to_return

    def apply_to_label_array(self, labels):
        """Applies re-mapping to a clustering output of labels array."""
        mapping_dict = dict(zip(self.to_values, self.from_values))
        return np.asarray([mapping_dict[i] for i in labels])


def inverse_transform_parameters(clusterer, scaler, covariance_type='diag', n_components: Optional[int] = None):
    # Cast the covariances to the right shape and extract the ones we need
    if covariance_type == 'spherical':
        pmra_cov = pmdec_cov = ra_cov = dec_cov = parallax_cov = clusterer.covariances_[:]

    elif covariance_type == 'diag':
        ra_cov = clusterer.covariances_[:, 0]
        dec_cov = clusterer.covariances_[:, 1]
        pmra_cov = clusterer.covariances_[:, 2]
        pmdec_cov = clusterer.covariances_[:, 3]
        parallax_cov = clusterer.covariances_[:, 4]

    elif covariance_type == 'full':
        ra_cov = clusterer.covariances_[:, 0, 0]
        dec_cov = clusterer.covariances_[:, 1, 1]
        pmra_cov = clusterer.covariances_[:, 2, 2]
        pmdec_cov = clusterer.covariances_[:, 3, 3]
        parallax_cov = clusterer.covariances_[:, 4, 4]

    elif covariance_type == 'tied':
        ra_cov = np.repeat(clusterer.covariances_[0, 0], n_components)
        dec_cov = np.repeat(clusterer.covariances_[1, 1], n_components)
        pmra_cov = np.repeat(clusterer.covariances_[2, 2], n_components)
        pmdec_cov = np.repeat(clusterer.covariances_[3, 3], n_components)
        parallax_cov = np.repeat(clusterer.covariances_[4, 4], n_components)

    else:
        raise ValueError("selected covariance_type not supported!")

    # Now, let's handle the means (rather a bit easier)
    means_array = scaler.inverse_transform(clusterer.means_)

    # And combine everything!
    parameters = {
        'lon': means_array[:, 0],
        'lon_std': np.sqrt(ra_cov * scaler.scale_[0] ** 2),
        'lat': means_array[:, 1],
        'lat_std': np.sqrt(dec_cov * scaler.scale_[1] ** 2),
        'pmlon': means_array[:, 2],
        'pmlon_std': np.sqrt(pmra_cov * scaler.scale_[2] ** 2),
        'pmlat': means_array[:, 3],
        'pmlat_std': np.sqrt(pmdec_cov * scaler.scale_[3] ** 2),
        'parallax': means_array[:, 4],
        'parallax_std': np.sqrt(parallax_cov * scaler.scale_[4] ** 2)
    }

    # Convert to a DataFrame and return
    parameters = pd.DataFrame(parameters)
    return parameters


default_gmm_kwargs = {
    'covariance_type': 'diag',
    'n_init': 1,
    'verbose': 0
}

default_partition_kwargs = {
    'constraints': [[None, None, 0.],
                    [5,    6,    700.],
                    [6,    7,    2500.]],
    'final_distance': np.inf,
    'parallax_sigma_threshold': 1.,
    'minimum_size': 5,
    'verbose': False
}


def run_gmm(data: np.ndarray,
            data_gaia: pd.DataFrame,
            scaler: Union[RobustScaler, StandardScaler],
            pixel_id: int,
            minimum_n_components: int = 1,
            stars_per_component: Union[int, list, tuple, np.ndarray] = 800,
            kwargs_for_partition: Optional[dict] = None,
            **kwargs_for_algorithm):
    """Function for running scikit-learn's implementation of Gaussian Mixture Models on data, aka
    sklearn.mixture.GaussianMixture.

    Args:
        data (np.ndarray): data to use.
        data_gaia (pd.DataFrame): the original cut and pre-processed DataFrame, which is required for dataset
            partitioning.
        scaler (sklearn.preprocessing RobustScaler or StandardScaler): the scaler used to scale the data. Necessary to
            inverse-transform the Gaussian mixtures later so that their parameters can be returned.
        pixel_id (int): the pixel id of the central HEALPix level 5 pixel in this field.
        stars_per_component (int): the number of stars to have per component. This is different to the base
            implementation (which specifies just n_components) as it was found to be much better to specify n_components
            in a data-driven way like this.
            Default: 800 (a good start for typical Gaia DR2 data)
        kwargs_for_partition (dict, optional): a dictionary of arguments to pass to the DataPartition to over-ride any
            defaults.
            Default: None
        **kwargs_for_algorithm: additional kwargs to pass to sklearn.mixture.GaussianMixture.

    Returns:
        labels (n_samples,), probabilities (n_samples,), means (n_clusters,) and std_deviations (n_clusters,).


    """
    gmm_kwargs = default_gmm_kwargs
    gmm_kwargs.update(kwargs_for_algorithm,)

    partition_kwargs = default_partition_kwargs
    if kwargs_for_partition is not None:
        partition_kwargs.update(kwargs_for_partition)

    # Setup the dataset partition and delete the data for memory efficiency
    partitioner = ocelot.cluster.DataPartition(
        data_gaia, pixel_id, n_stars_per_component=stars_per_component, **partition_kwargs)
    partitioner.delete_data()
    gc.collect()

    # Blank lists of things to save
    labels = []
    probabilities = []
    mixture_parameters = []

    # Clusterer time!
    last_highest_label = 0
    for i_partition in range(partitioner.total_partitions):
        # Tell the user what the fuck is going on!!!
        sys.stdout.write(f"\r    partition {i_partition+1} of {partitioner.total_partitions}")
        sys.stdout.flush()

        # Grab the partition and the number of stars
        partition = partitioner.get_partition(i_partition, return_data=False)
        n_components = partitioner.get_n_components(minimum_n_components=minimum_n_components)

        # Make a new clusterer and fit it
        clusterer = GaussianMixture(n_components=n_components, **gmm_kwargs)
        labels.append(clusterer.fit_predict(data[partition]))
        probabilities.append(np.max(clusterer.predict_proba(data[partition]), axis=1))

        # We also want parameters of the mixtures that we can use for cutting later in the original co-ordinate space
        mixture_parameters.append(inverse_transform_parameters(
            clusterer, scaler, covariance_type=gmm_kwargs['covariance_type'], n_components=n_components))

        # And finally, we want to work out whether or not our friend is even valid
        validities, ra, dec = partitioner.test_if_in_current_partition(
            mixture_parameters[i_partition]['lon'], mixture_parameters[i_partition]['lat'],
            mixture_parameters[i_partition]['parallax'], return_ra_dec=True)

        # And add all this information to the mixture parameters!
        mixture_parameters[i_partition]['ra'] = ra
        mixture_parameters[i_partition]['dec'] = dec
        mixture_parameters[i_partition]['valid_pixel'] = validities
        mixture_parameters[i_partition]['partition'] = i_partition
        mixture_parameters[i_partition]['cluster_label'] = np.arange(last_highest_label,
                                                                     last_highest_label + ra.shape[0])

        # We have to be a little bit more careful with the number of stars per mixture, as sometimes, mixtures won't
        # contain any stars at all.
        indexes_with_stars, counts = np.unique(labels[i_partition], return_counts=True)
        mixture_parameters[i_partition]['n_stars'] = 0
        mixture_parameters[i_partition].loc[indexes_with_stars, 'n_stars'] = counts

        # Lastly, change up the labels for the clusters to carry on consecutively from the last round
        labels[i_partition] += last_highest_label
        last_highest_label = np.max(labels[i_partition]) + 1

    # Last update to the user. Important that it has the \n, which means we finally get a goddamn new line
    sys.stdout.write(f"\r    final partition is complete =)\n")
    sys.stdout.flush()

    return labels, probabilities, mixture_parameters, partitioner


class GMMPipeline(ClusteringAlgorithm):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 pixel_ids: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, List[Path]]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 verbose: bool = True,
                 user_kwargs: Optional[Union[List[dict], Tuple[dict]]] = None,):
        """Looking to run GMMs on a large selection of fields? Then look no further! This pipeline has you covered.

        Args:
            names (list-like): list of names of fields to run on to save.
            pixel_ids (list-like): list of HEALPix level 5 pixel ids of the central pixel. These are required by the
                partitioning scheme!
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

        super().__init__(run_gmm,
                         user_kwargs,
                         names,
                         input_dirs,
                         input_patterns=input_patterns,
                         output_dirs=output_dirs,
                         verbose=verbose,
                         required_input_keys=['cut', 'rescaled', 'scaler'],
                         required_output_keys=['partitioned_results',
                                               'mixture_parameters', 'partition_plots', 'times'],
                         extra_returned_info=['mixture_parameters', 'partitioner'],

                         # Saving of baby data gaia views isn't supported at all, since we implement our own version of
                         # save_clustering_result below anyway
                         max_cluster_size_to_save=0, )

        # Overwrite a couple of superclass things because GMMs are fucking weird - they have lots of extra inputs that
        # I want to make super sure are in the correct order, and also require pixel_ids (a hard coded special case)
        self.pixel_ids = pixel_ids
        self.extra_input_names = ['cut', 'scaler', 'pixel_id']

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
        mixture_parameters = algorithm_return[2]
        partitioner: ocelot.cluster.DataPartition = algorithm_return[3]
        n_partitions = len(labels)

        # Save the labels and probabilities arrays
        for a_partition in range(n_partitions):
            # We'll want to index them with the indexes into the original partition
            partition = partitioner.get_partition(a_partition, return_data=False)

            # And save it all!
            pd.DataFrame({'index_into_cut_data': partition.nonzero()[0],
                          'labels': labels[a_partition],
                          'probabilities': probabilities[a_partition]}
                         ).to_feather(
                self.output_paths['partitioned_results'] / Path(f"{joined_name}_part{a_partition}_results.feather"))

        # Join all the mixture parameters together
        final_cluster_info = pd.concat(mixture_parameters, ignore_index=True)

        final_cluster_info.to_csv(
            self.output_paths['mixture_parameters'] / Path(f"{joined_name}_mixture_parameters.csv"), index=False)

        # Lastly, let's make a plot of the partitions for future reference.
        partitioner.plot_partition_bar_chart(
            figure_title=f'Field {field_name}, run {run_name}',
            save_name=str((self.output_paths['partition_plots'] / Path(f"{joined_name}_partition.png")).resolve()),
            show_figure=False,
            desired_size=partitioner.minimum_partition_size
        )
        plt.close('all')

        # _save_clustering_time expects an n_clusters value, so we return it here (although this includes the many,
        # many clusters that aren't real in the GMM method)
        return final_cluster_info.shape[0]


default_cuts = {
    'proper_motion': tcg_proper_motion_cut,
    'size': size_cut,
}

default_cut_parameters = {
    'min_size': 10,
    'max_size': np.inf
}


class GMMPostProcessor(Pipeline):
    def __init__(self,
                 names: Union[list, tuple, np.ndarray],
                 input_dirs: Dict[str, Union[Path, list]],
                 input_patterns: Optional[Dict[str, str]] = None,
                 output_dirs: Optional[Dict[str, Path]] = None,
                 cuts: Optional[dict] = None,
                 cut_parameters: Optional[dict] = None,
                 max_cluster_size_to_save: int = 10000,
                 verbose: bool = True,):
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

            cuts (dict): cuts to pass to ocelot.preprocess.cut_dataset.
                Default: None
            # TODO: update me!
            max_cluster_size_to_save (int): maximum cluster size to save baby_data_gaia views for. If 0 or -ve, then we
                also don't calculate any statistics for the cluster, and just return the labels (+probabilities).
                Default: 10000
            verbose (bool): whether or not to run in verbose mode with informative print statements.
                Default: True

        """
        # We cast names as a list, because this class requires it to be one to use the .index method later
        names = list(names)

        super().__init__(names,
                         input_dirs,
                         input_patterns=input_patterns, output_dirs=output_dirs, verbose=verbose,
                         required_input_keys=['cut', 'partitioned_results', 'mixture_parameters',],
                         required_output_keys=['labels', 'probabilities',
                                               'cluster_data', 'cluster_list',],
                         check_input_shape=False)

        # Write the cuts to the class
        self.cuts = default_cuts
        if cuts is not None:
            self.cuts.update(cuts)

        self.cut_parameters = default_cut_parameters
        if cut_parameters is not None:
            self.cut_parameters.update(cut_parameters)

        # Pre-process some info about the input files for later use.
        all_partitioned_results_file_stems = [x.stem.rsplit(sep='_') for x in self.input_paths['partitioned_results']]
        self.partitioned_results_field_and_run_names = np.asarray(
            ['_'.join(x[0:2]) for x in all_partitioned_results_file_stems])
        self.partitioned_results_partition_numbers = np.asarray(
            [int((x[2])[4:]) for x in all_partitioned_results_file_stems])  # We assume they all start with 'part'

        # Check that the cut_data input file has the same shape as the names
        if len(self.input_paths['cut']) != len(self.names):
            raise ValueError("there must be the same number (and ordering) between the cut_data and the input paths for"
                             " cut data, but they don't have the same shape!")

        self._max_cluster_size_for_stats = max_cluster_size_to_save

        if self.verbose:
            print("GMM preprocessing pipeline initialised!")

    def _get_all_run_data(self, mixture_parameters_path: Path):
        """Backtracks using the current mixture_parameters_path to find all partitioned results files and the correct
        cut_data file."""
        # First, let's get the field_name and run_name from the file
        split_up_path = str(mixture_parameters_path.stem).rsplit(sep='_')
        field_name, run_name = split_up_path[0:2]

        # Then, let's find the correct cut_data for this purpose
        cut_data_path = self.input_paths['cut'][self.names.index(field_name)]

        # And then let's find all the individual partitioned results too
        to_match = f"{field_name}_{run_name}"
        matched_partitions = (self.partitioned_results_field_and_run_names == to_match).nonzero()[0]
        partitions_numbers = self.partitioned_results_partition_numbers[matched_partitions]
        partitions_paths = [self.input_paths['partitioned_results'][x] for x in matched_partitions]

        # Ok, that's the file stuff done. Now let's read it all in!
        mixture_parameters = self._open(mixture_parameters_path)
        expected_partitions = np.unique(mixture_parameters['partition'])

        # Make sure that we have the same number of partitions in the mixture results file and the individual stuff
        # we've read in
        n_expected_partitions = len(expected_partitions)
        n_partitions = len(partitions_numbers)
        if not np.all(np.isin(partitions_numbers, expected_partitions)) or n_expected_partitions != n_partitions:
            raise ValueError(f"wrong number or type of partitioned_results files were found! We expected to find "
                             f"{n_expected_partitions} based on the partitions in the mixture parameters DataFrame, "
                             f"but only found {n_partitions} corresponding partitioned_results files. "
                             f"\nWe expected the following partition numbers: \n{expected_partitions} "
                             f"\nBut only found the following files: \n{partitions_numbers} ")

        # Read in the individual partitioned results files and append them to a dictionary based on what their number
        # is
        partitioned_results = {}
        for a_partition_path, a_partition_number in zip(partitions_paths, partitions_numbers):
            partitioned_results[a_partition_number] = self._open(a_partition_path)

        # And lastly, we want the cut dataframe too
        data_gaia = self._open(cut_data_path)

        return data_gaia, mixture_parameters, partitioned_results, field_name, run_name

    def _apply_cuts_to_one_field(self, mixture_parameters, field_name, run_name):
        """Cycles over a field, working out which clusters are and aren't crap."""
        # First off, let's make a new DataFrame to hold all information we have on it
        joined_name = f"{field_name}_{run_name}"
        cluster_statistics_frame = pd.DataFrame({
            'field': field_name,
            'run': run_name,
            'cluster_label': mixture_parameters['cluster_label'],
            'cluster_id': [joined_name + f"_{x}" for x in mixture_parameters['cluster_label']],
            'partition': mixture_parameters['partition'],
            'n_stars': mixture_parameters['n_stars'],
            'valid_pixel': mixture_parameters['valid_pixel'],
        })

        # Now, for every specified cut type, we'll add it to the cluster statistics frame
        validity_strings = ['valid_pixel']
        for a_cut_key in self.cuts:
            a_cut_name = f"valid_{a_cut_key}"
            cluster_statistics_frame[a_cut_name] = self.cuts[a_cut_key](
                mixture_parameters, self.cut_parameters)
            validity_strings.append(a_cut_name)

        # Finally, work out a total score of whether or not this cluster is valid
        cluster_statistics_frame['valid_total'] = np.all(cluster_statistics_frame[validity_strings].to_numpy(), axis=1)

        valid_clusters = np.asarray(cluster_statistics_frame['valid_total']).nonzero()[0]

        return cluster_statistics_frame, valid_clusters

    def apply(self, start: int = 0):
        """Applies the post-processing to the selected clusters.

        Args:
            start (int): the cluster number in self.names to start with.
                Default: 0

        """
        completed_steps = start
        total_steps = len(self.input_paths['mixture_parameters'])

        iteration_start = datetime.datetime.now()

        # Loop over all the different fields
        for a_mixture_parameters_path in self.input_paths['mixture_parameters']:
            if self.verbose:
                print(f"-- {datetime.datetime.today()}")
                print(f"Performing GMM post-processing on field & run number {completed_steps+1} of {total_steps}")

            # Read in all the lovely data frens
            data_gaia, mixture_parameters, partitioned_results, field_name, run_name = \
                self._get_all_run_data(a_mixture_parameters_path)

            joined_name = f"{field_name}_{run_name}"

            # Identify which clusters are good and start off our cluster_list DataFrame to save
            data_cluster, valid_clusters = self._apply_cuts_to_one_field(mixture_parameters, field_name, run_name)

            # Also add all the ocelot columns to this (gonna be a fuck ton of NaNs, sorry future me but this is what
            # consistency looks like with your past work xoxo)
            data_cluster = data_cluster.join(
                _blank_statistics_dataframe.drop(['field', 'run', 'cluster_label', 'cluster_id', 'n_stars'],
                                                 axis='columns'))

            if self.verbose:
                print(f"  found {len(valid_clusters)} valid clusters")
                print(f"  calculating relevant labels for them all")

            # Iterate over good clusters, opening up their partitioned results files and finding the relevant stars
            # in the cut DataFrame
            labels = np.full(data_gaia.shape[0], -1, dtype=int)
            probabilities = np.zeros(data_gaia.shape[0])
            for a_cluster in valid_clusters:

                # Find all stars that match the label of this cluster
                label_to_find, partition_in_question = data_cluster.loc[a_cluster, ['cluster_label', 'partition']]
                matched_stars = partitioned_results[partition_in_question].loc[
                    partitioned_results[partition_in_question]['labels'] == label_to_find]

                # See which stars this is a more probable clustering for than all we've seen previously
                boolean_array_into_matched_stars = (probabilities[matched_stars['index_into_cut_data'].to_numpy()]
                                                    < matched_stars['probabilities'].to_numpy())
                indexes_into_cut_data_to_change = \
                    matched_stars.loc[boolean_array_into_matched_stars, 'index_into_cut_data'].to_numpy()

                # Hence, save these labels and probabilities
                labels[indexes_into_cut_data_to_change] = \
                    matched_stars.loc[boolean_array_into_matched_stars, 'labels']
                probabilities[indexes_into_cut_data_to_change] = \
                    matched_stars.loc[boolean_array_into_matched_stars, 'probabilities']

            if self.verbose:
                print(f"  now calculating parameters with ocelot")

            # Do a more "normal" calculation loop of parameters inferred precisely by ocelot
            parameter_frames_to_concatenate = []
            for a_cluster in valid_clusters:
                # Name the cluster!!!!
                cluster_id = joined_name + f"_{a_cluster}"

                # Grab all the relevant stars and stats
                good_stars = labels == a_cluster
                baby_data_gaia = data_gaia.loc[good_stars].reset_index(drop=True)

                baby_probabilities = probabilities[good_stars]

                # Run ocelot and pop it into the list of parameter frames
                a_series = pd.Series(ocelot.calculate.all_statistics(
                    baby_data_gaia, membership_probabilities=baby_probabilities))

                cluster_in_data_cluster = data_cluster.index[data_cluster['cluster_label'] == a_cluster].to_numpy()[0]
                data_cluster.loc[cluster_in_data_cluster, a_series.index] = a_series

                # Save the small data_gaia view
                if a_series['n_stars'] < self._max_cluster_size_for_stats:

                    baby_data_gaia['probability'] = baby_probabilities

                    baby_data_gaia.to_csv(
                        self.output_paths['cluster_data'] / Path(f"{cluster_id}_cluster_data.csv"))

                del baby_data_gaia
                gc.collect()

            if self.verbose:
                print("  and saving the results!")

            # Save the output!
            pd.DataFrame({'labels': labels}).to_feather(
                self.output_paths['labels'] / Path(f"{joined_name}_labels.feather"))

            pd.DataFrame({'probabilities': probabilities}).to_feather(
                self.output_paths['probabilities'] / Path(f"{joined_name}_probs.feather"))

            data_cluster.to_csv(self.output_paths['cluster_list'] / Path(f"{joined_name}_cluster_list.csv"),
                                index=False)

            completed_steps += 1
            if self.verbose:
                print_itertime(iteration_start, completed_steps, total_steps)
