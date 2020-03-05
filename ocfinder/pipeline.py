"""A set of functions for general tasks I like to do. Each one is designed to work on hilariously large scales and
datasets without too many tears."""

import ocelot
import numpy as np
import pickle
import json
import pandas as pd
import gc

from pathlib import Path


def _open(path: Path, mode='pickle'):
    """Cheeky function to read in Gaia data from a pickle file or a JSON."""
    if mode == 'pickle':
        with open(str(path.resolve()), 'rb') as handle:
            return pickle.load(handle)
    elif mode == 'json':
        return pd.read_json(path)
    else:
        raise ValueError("Specified input_mode not supported! Must be either pickle or json.")


default_rescale_kwargs = {
    'columns_to_rescale': ('lon', 'lat', 'pmlon', 'pmlat', 'parallax'),
    'column_weights': (1., 1., 1., 1., 1.),
    'scaling_type': 'robust',
    'concatenate': True,
    'return_scaler': True,
}


def preprocess(input_dir: Path, output_dir: Path, cuts: dict, centers,
               rescale_kwargs=None, verbose=True, input_mode='pickle'):
    """Starts an autonomous pre-processing pipeline. To cut, re-center and re-scale data and save it locally."""
    if verbose:
        print("Preprocessing pipeline initialised!")

    # Handle default args
    final_rescale_kwargs = default_rescale_kwargs
    if rescale_kwargs is not None:
        for a_kwarg in rescale_kwargs.keys():
            final_rescale_kwargs[a_kwarg] = rescale_kwargs[a_kwarg]

    # Find all the clusters in the directory
    if input_mode == 'pickle':
        paths_to_clusters = Path(input_dir).glob("*.pickle")
        extension_length = 7
    elif input_mode == 'json':
        paths_to_clusters = Path(input_dir).glob("*.json")
        extension_length = 5
    else:
        raise ValueError("Specified input_mode not supported! Must be either pickle or json.")

    # Cycle over each cluster, applying all of the required pre-processing steps
    for a_path, a_center in zip(paths_to_clusters, centers):
        name = str(a_path).rsplit("/")[-1][:-extension_length]
        if verbose:
            print(f"Working on field {name}...")
            print(f"  cutting dataset")

        data_gaia = _open(a_path, mode=input_mode)

        # Apply cuts and scaling
        data_gaia = ocelot.cluster.cut_dataset(data_gaia, cuts)

        if verbose:
            print(f"  re-centering dataset")
        data_gaia = ocelot.cluster.recenter_dataset(data_gaia, center=a_center)

        if verbose:
            print(f"  re-scaling dataset")
        data_rescaled, scaler = ocelot.cluster.rescale_dataset(data_gaia, **final_rescale_kwargs)

        if verbose:
            print(f"  saving them both!")

        data_gaia.to_json(output_dir / Path(f"{name}_cut.json"))

        with open(output_dir / Path(f"{name}_scaler.pickle"), 'wb') as handle:
            pickle.dump(scaler, handle, pickle.HIGHEST_PROTOCOL)

        with open(output_dir / Path(f"{name}_rescaled.json"), 'w') as json_file:
            json.dump(data_rescaled, json_file)

        del data_gaia, scaler, data_rescaled
        gc.collect()

        if verbose:
            print(f"  both saved successfully!")

    if verbose:
        print(f"  pre-processing is complete!")





def upmask():
    pass
