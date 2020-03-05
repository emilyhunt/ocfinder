"""Functions for saving of found open cluster data."""

import ocelot
import numpy as np
import pandas as pd
import gc
import json

from typing import Optional
from pathlib import Path


def save_clustering_result(data_gaia: pd.DataFrame,
                           labels: np.ndarray,
                           name: str,
                           output_dir: Path,
                           probabilities: Optional[np.ndarray] = None,
                           save_data_gaia_view: bool = True):
    """Saves the result of a clustering analysis.

    Args:
        data_gaia (pd.DataFrame): Gaia data. You know the drill by now.
        labels (np.ndarray): sklearn-style clustering label output into data_gaia, where -1 is noise.
        name (str): name of the clustering result currently being studied, e.g. "0042_dbscan_acg"
        output_dir (pathlib.Path): directory to output results to.
        probabilities (optional, np.ndarray): probabilities to save too, assuming this is a method that has them.
            Default: None
        save_data_gaia_view (bool): whether or not to save a mini data_gaia. Not essential as it can be made manually
            later with the labels array! Useful to turn off if the algorithm's output becomes too noisy to save HD space
            Default: True

    Returns:
        happiness. joy. fulfillment. aka None

    """
    # Stuff we need
    all_cluster_indices = np.unique(labels)
    all_cluster_indices = all_cluster_indices[all_cluster_indices != -1]

    parameter_frames_to_concatenate = []

    for a_label in all_cluster_indices:
        # Name the cluster!!!!
        cluster_id = name + f"_{a_label}"

        # Find all the stars that match and make a baby data_gaia, including probabilities
        baby_data_gaia = data_gaia.loc[labels == a_label].reset_index(drop=True)

        # Calculate all statistics with ocelot
        a_dict = {'id': cluster_id}
        a_dict.update(ocelot.calculate.all_statistics(baby_data_gaia, membership_probabilities=probabilities))
        parameter_frames_to_concatenate.append(pd.DataFrame(a_dict, index=[0]))

        # Save the small data_gaia view
        if save_data_gaia_view:
            baby_data_gaia.to_json(output_dir / Path(f"{cluster_id}_data.json"))

        del data_gaia
        gc.collect()

    # Save the labels and probabilities arrays
    with open(output_dir / Path(f"{name}_labels.json"), 'w') as json_file:
        json.dump(labels, json_file)

    if probabilities is not None:
        with open(output_dir / Path(f"{name}_probs.json"), 'w') as json_file:
            json.dump(labels, json_file)

    # Concatenate into a DataFrame and save the parameters
    pd.concat(parameter_frames_to_concatenate, ignore_index=True).to_csv(output_dir / Path(f"{name}_clusters.csv"))
