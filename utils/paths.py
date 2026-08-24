import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUTS_PATH = f"{BASE_PATH}/outputs"
EXPERIMENTS_ASSETS_PATH = f"{BASE_PATH}/experiments_assets"

MANIFEST_DICTIONARIES = f"{EXPERIMENTS_ASSETS_PATH}/dictionaries/manifest_dictionaries.json"
PATHS_DICTIONARIES = f"{EXPERIMENTS_ASSETS_PATH}/dictionaries/paths_dictionaries.json"
MANIFEST_DATASETS = f"{EXPERIMENTS_ASSETS_PATH}/datasets/manifest_datasets.json"
PATHS_DATASETS = f"{EXPERIMENTS_ASSETS_PATH}/datasets/paths_datasets.json"
MANIFEST_ARRAYS = f"{EXPERIMENTS_ASSETS_PATH}/arrays/manifest_arrays.json"

MANIFEST_FRAMEWORKS = f"{OUTPUTS_PATH}/spectrums/manifest_frameworks.json"
MANIFEST_SPECTRUMS = f"{OUTPUTS_PATH}/spectrums/manifest_spectrums.json"
MANIFEST_METRICS = f"{OUTPUTS_PATH}/metrics/manifest_metrics.json"

PATHS_SPECTRUMS = f"{OUTPUTS_PATH}/spectrums/paths_spectrums.json"
PATHS_METRICS = f"{OUTPUTS_PATH}/metrics/paths_metrics.json"
