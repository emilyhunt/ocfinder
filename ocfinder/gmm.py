"""Functions for running Gaussian mixture models on a field."""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, StandardScaler

from typing import Union, Optional


def tcg_proper_motion_cut(mean, std, cut_parameters):
    # Todo fstring
    return np.where(mean['parallax'] > 0.6703, 5 * np.sqrt(2) / 4.74 * mean['parallax'], 1)


def radius_cut(mean, std, cut_parameters):
    # Todo fstring
    distances = np.clip(1000 / mean['parallax'], 0, cut_parameters['max_distance'])
    return np.arctan(cut_parameters['max_radius'] / distances) * 180 / np.pi


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

    std = {}
    std['ra'] = np.sqrt(ra_cov * scaler.scale_[0] ** 2)
    std['dec'] = np.sqrt(dec_cov * scaler.scale_[1] ** 2)
    std['pmra'] = np.sqrt(pmra_cov * scaler.scale_[2] ** 2)
    std['pmdec'] = np.sqrt(pmdec_cov * scaler.scale_[3] ** 2)
    std['parallax'] = np.sqrt(parallax_cov * scaler.scale_[4] ** 2)
    std = pd.DataFrame(std)

    # Now, let's handle the means (rather a bit easier)
    means_array = scaler.inverse_transform(clusterer.means_)
    mean = {}
    mean['ra'] = means_array[:, 0]
    mean['dec'] = means_array[:, 1]
    mean['pmra'] = means_array[:, 2]
    mean['pmdec'] = means_array[:, 3]
    mean['parallax'] = means_array[:, 4]
    mean = pd.DataFrame(mean)

    return mean, std


default_gmm_kwargs = {
    'covariance_type': 'diag',
    'n_init': 1,
}


def run_gmm(data: np.ndarray, scaler: Union[RobustScaler, StandardScaler],
            stars_per_component: int = 800,
            **kwargs_for_algorithm):
    """Function for running scikit-learn's implementation of Gaussian Mixture Models on data, aka
    sklearn.mixture.GaussianMixture.

    Args:
        data (np.ndarray): data to use.
        scaler (sklearn.preprocessing RobustScaler or StandardScaler): the scaler used to scale the data. Necessary to
            inverse-transform the Gaussian mixtures later so that their parameters can be returned.
        stars_per_component (int): the number of stars to have per component. This is different to the base
            implementation (which specifies just n_components) as it was found to be much better to specify n_components
            in a data-driven way like this.
            Default: 800 (a good start for typical Gaia DR2 data)
        **kwargs_for_algorithm: additional kwargs to pass to sklearn.mixture.GaussianMixture.

    Returns:
        labels (n_samples,), probabilities (n_samples,), means (n_clusters,) and std_deviations (n_clusters,).


    """
    gmm_kwargs = default_gmm_kwargs
    gmm_kwargs.update(kwargs_for_algorithm,
                      n_components=int(np.round(data.shape[0] / stars_per_component)))

    # Clusterer time!
    clusterer = GaussianMixture(**gmm_kwargs)
    labels = clusterer.fit_predict(data)
    probabilities = clusterer.predict_proba(data)

    # We also want parameters of the mixtures that we can use for cutting later, but in the original co-ordinate space
    means, std_deviations = inverse_transform_parameters(
        clusterer, scaler, covariance_type=gmm_kwargs['covariance_type'], n_components=gmm_kwargs['n_components'])

    return labels, probabilities, means, std_deviations


def cut_gmm_results():
    """Function for cutting Gaussian Mixture Model (gmm) results using the clusters themselves."""
    # Todo this function once I have some results!
    pass


