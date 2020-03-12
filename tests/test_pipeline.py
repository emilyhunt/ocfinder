"""Functions for integration testing of the pipeline superclass. Designed for use with pytest!"""

import ocfinder.pipeline
import ocfinder.hdbscan
import ocfinder.dbscan
import ocfinder.upmask
import ocfinder.gmm

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

path_to_blanco_1 = Path('./test_data/blanco_1_gaia_dr2_gmag_18_cut.pickle')


def test_pipeline():
    """Tests the pipeline superclass and its _open protected method."""
    input_dirs = {'data': path_to_blanco_1, 'also_data': path_to_blanco_1}
    required_input_keys = ['data', 'also_data']
    required_output_keys = None

    # Make a new pipeline
    my_pipeline = ocfinder.pipeline.Pipeline(['Blanco_1'],
                                             input_dirs,
                                             check_input_shape=True,
                                             required_input_keys=required_input_keys,
                                             required_output_keys=required_output_keys)

    # Can we open a file autonomously?
    data_gaia = my_pipeline._open(my_pipeline.input_paths['data'][0])

    # Test the shape
    assert data_gaia.shape == (14785, 28)

    return my_pipeline, data_gaia


def test_preprocessor():
    """Tests the preprocessing pipeline."""
    input_dirs = {'data': path_to_blanco_1}
    output_dirs = {'cut': Path('./test_preprocessor_output'),
                   'scaler': Path('./test_preprocessor_output'),
                   'rescaled': Path('./test_preprocessor_output')}

    preprocessor = ocfinder.pipeline.Preprocessor(['Blanco_1'],
                                                  input_dirs,
                                                  output_dirs=output_dirs,
                                                  cuts={'parallax': [1, np.inf]},
                                                  centers=[[0.885, -30.]])

    preprocessor.apply()

    data_cut = pd.read_feather(output_dirs['cut'] / Path('Blanco_1_cut.feather'))

    assert data_cut.shape == (7378, 32)

    return preprocessor, data_cut


def test_hdbscan():
    """Tests the HDBSCAN usage of the ClusteringAlgorithm superclass et al."""
    input_dirs = {'cut': Path('./test_preprocessor_output/Blanco_1_cut.feather'),
                  'rescaled': Path('./test_preprocessor_output/Blanco_1_rescaled.feather')}
    output_dirs = {'labels': Path('./test_hdbscan_output'),
                   'probabilities': Path('./test_hdbscan_output'),
                   'cluster_data': Path('./test_hdbscan_output'),
                   'cluster_list': Path('./test_hdbscan_output'),
                   'times': Path('./test_hdbscan_output')}

    industrial_strength_clustering_pipeline = ocfinder.hdbscan.HDBSCANPipeline(
        ['Blanco_1'],
        input_dirs,
        output_dirs=output_dirs,
        user_kwargs=[{'min_cluster_size': 40, 'min_samples': 10},
                     {'min_cluster_size': 80, 'min_samples': 10}]
    )

    industrial_strength_clustering_pipeline.apply()


def test_dbscan():
    """Tests the DBSCAN usage of the ClusteringAlgorithm superclass et al."""
    # Todo
    pass


def test_gmm():
    """Tests the Gaussian Mixture Model (GMM) usage of the ClusteringAlgorithm superclass et al."""
    # Todo
    pass


def test_upmask():
    """Tests the UPMASK usage of the ClusteringAlgorithm superclass et al."""
    # Todo
    pass


def test_plotting():
    input_dirs = {'cut': Path('./test_preprocessor_output/Blanco_1_cut.feather'),
                  'labels': Path('./test_hdbscan_output'),
                  'probabilities': Path('./test_hdbscan_output'),
                  'cluster_list': Path('./test_hdbscan_output'),
                  'times': Path('./test_hdbscan_output/runtimes.csv')}
    input_patterns = {'labels': '*_labels.feather',
                      'probabilities': '*_probs.feather',
                      'cluster_list': '*_cluster_list.csv',}
    output_dirs = {'plots': Path('./test_plots')}

    kwargs_for_plotting_algorithm = {
        'cmd_plot_y_limits': [8, 18],
        'dpi': 300,
        'cluster_marker_radius': (2., 2., 2.),
    }

    industrial_strength_plotting_pipeline = ocfinder.pipeline.ResultPlotter(
        ['Blanco_1'],
        input_dirs,
        input_patterns=input_patterns,
        output_dirs=output_dirs,
        **kwargs_for_plotting_algorithm
    )

    # return industrial_strength_plotting_pipeline

    industrial_strength_plotting_pipeline.apply(
        plot_clusters_individually=True,)  # threshold=0.05, threshold_comparison='>=', threshold_key='persistences')

    return industrial_strength_plotting_pipeline

if __name__ == '__main__':
    # pipe, gaia = test_pipeline()
    # pipe, gaia = test_preprocessor()
    # test_hdbscan()
    pipe = test_plotting()