import os
from pathlib import Path
import torch
import math


from utils.assets_management_utils import generate_8char_tag, key_management
from frameworks.frameworks_wrapper import wrapper, FRAMEWORKS_CANONICAL_SIGNATURES

from utils.paths import PATHS_DICTIONARIES, PATHS_DATASETS, MANIFEST_DATASETS
from utils.paths import OUTPUTS_PATH, MANIFEST_FRAMEWORKS, MANIFEST_SPECTRUMS, MANIFEST_METRICS, PATHS_SPECTRUMS, PATHS_METRICS



def k_largest_peaks(x, k):
    """
    Outputs the indices of the k largest peaks in a 1D spectrum tensor x in order of descending magnitude.
    Inputs:
        x (torch.Tensor): A 1D tensor containing the signal.
        k (int): The number of largest peaks to return.
    Outputs:
        torch.Tensor: Indices of the k largest peaks in descending order of magnitude.
    """
    # Initialize peak mask
    is_peak = torch.zeros(x.shape, dtype=torch.bool)

    # Check interior points
    is_peak[1:-1] = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])

    # Check first and last points
    if len(x) > 1:
        if x[0] > x[1]:
            is_peak[0] = True
        if x[-1] > x[-2]:
            is_peak[-1] = True
    else:
        # If x has only one element, consider it a peak
        is_peak[0] = True

    # Get the values and indices of all peaks
    peak_values = x[is_peak]
    peak_indices = torch.where(is_peak)[0]

    # Sort peaks in descending order
    sorted_indices = torch.argsort(peak_values, descending=True)

    # Get the indices of the k largest peaks
    k = min(k, len(sorted_indices))
    top_k_indices = sorted_indices[:k]

    # Get the original locations of the k largest peaks
    peak_locs = peak_indices[top_k_indices]

    return peak_locs



def spectrum_exists(framework, dataset_tag, **kwargs):

    """
    Checks if a spectrum resulting from a framework with a specific configuration on a given dataset already exists in the outputs/spectrums directory. Returns the spectrum state (True if it exists, False otherwise), the spectrum setup, the framework configuration, the spectrum tag, and the framework tag.

    Inputs:
        framework (str): The name of the framework to evaluate. Options are 'LASSO', 'L1L2', 'TL1', 'MCP','SR-LASSO', 'SR-TL1', 'SBL'.
        dataset_tag (str): The tag of the test dataset. The list of available dataset tags can be found in the manifest_datasets.json file in the experiments_assets/datasets directory.
        **kwargs: Additional keyword arguments specifying the framework configuration. See the FRAMEWORKS_CANONICAL_SIGNATURES dictionary in frameworks/frameworks_wrapper.py for the expected parameters for each framework.

    Outputs:
        spectrum_state (bool): True if the spectrum exists, False otherwise.
        spectrum_setup (dict): A dictionary containing the framework configuration and the dataset tag.
        framework_configuration (dict): A dictionary containing the framework configuration. The items are ordered according to the canonical order specified in FRAMEWORKS_CANONICAL_SIGNATURES. 
        spectrum_tag (str): The unique identifier for the spectrum resulting from the specific framework configuration and dataset tag.
        framework_tag (str): The unique identifier for the framework configuration.
    """

    # 2. Get the predefined order for the current framework (fallback to alphabetical if missing)
    canonical_keys = FRAMEWORKS_CANONICAL_SIGNATURES.get(framework, [])
    
    spectrum_setup= {}
    framework_configuration = {}
    # Always put framework first
    framework_configuration["framework"] = framework
    
    # Build the dict using the exact canonical order specified
    for key in canonical_keys:
        if key in kwargs:
            if isinstance(kwargs[key], int) and key not in ["iterations", "outer_iterations"]:
                framework_configuration[key] = float(kwargs[key])
            else:
                framework_configuration[key] = kwargs[key]


    spectrum_setup.update(framework_configuration)
    spectrum_setup["dataset_tag"] = dataset_tag
    framework_tag = generate_8char_tag(framework_configuration)
    spectrum_tag = generate_8char_tag(spectrum_setup)

    spectrum_state = key_management(MANIFEST_SPECTRUMS, spectrum_tag, mode="query")

    return spectrum_state, spectrum_setup, framework_configuration, spectrum_tag, framework_tag




def metric_estimate(metric, spectrum, angular_grid, angles, amplitudes, angular_bins_threshold=2, 
                    amplitude_threshold=0.4, false_alarm_threshold=0.01):
    """
    Evaluates the performance with respect to a metric of framework-estimated spectrum on an angular grid using the corresponding ground truth angles and amplitudes. The metric can be one of 'detection_rate', 'rmse', 'false_alarm_rate'.

        Inputs:
            metric (str): the metric to compute. Options are 'detection_rate', 'rmse', 'false_alarm_rate'.
            spectrum (torch.Tensor): the estimated spectrum of shape (N,).   
            angular_grid (torch.Tensor): the angular grid corresponding to the spectrum of shape (N,).
            angles (torch.Tensor): the ground truth angles of shape (K,).
            amplitudes (torch.Tensor): the ground truth amplitudes of shape (K,).
            angular_bins_threshold (int): the number of bins in the angular grid that correspond to the angular threshold for peak detection.
            amplitude_threshold (float): the amplitude threshold for peak detection as a fraction of the ground truth amplitude
            false_alarm_threshold (float): the threshold for false alarm rate metric

    Outputs:
        metric_value: float (or nan for no detections), the computed metric value.
    """
    N = len(spectrum)

    amplitudes = torch.abs(amplitudes)
    angles = torch.real(angles)

    # Get the k largest peaks in the estimated spectrum
    spec_pk_supp = k_largest_peaks(torch.abs(spectrum), N // 2)
    spec_pk_amp = torch.abs(spectrum[spec_pk_supp])

    spec_pk_supp_angles = angular_grid[spec_pk_supp]
    
    angular_resolution = angular_grid[1] - angular_grid[0]
    angular_grid_fov = angular_grid[-1] - angular_grid[0] 

    angular_threshold = angular_bins_threshold * angular_resolution

    # Initialize detection and error tensors of length equal to the number of peaks in the ground truth spectrum
    detections = torch.zeros_like(angles, dtype=torch.float64)
    errors = torch.zeros_like(angles, dtype=torch.float64)

    def remove_elements(A, B):
        # Create a mask where elements in A that are not in B are True
        mask = ~torch.isin(A, B)
        return A[mask]

    # Stores the set of candidate peaks that have already been ascribed to a ground truth peak
    assigned = torch.tensor([])

    for k in range(len(angles)):
        # Find the peaks that are within the angular threshold of the kth ground truth peak and have an amplitude above the amplitude threshold relative to the kth ground truth peak
        dist1 = torch.abs(spec_pk_supp_angles - angles[k])
        dist1 = torch.min(dist1, angular_grid_fov - dist1)
        detected_indices1 = torch.where(dist1 <= angular_threshold)[0]
        detected_indices2 = torch.where(spec_pk_amp[detected_indices1] >= amplitude_threshold * amplitudes[k])[0]

        detected_indices = detected_indices1[detected_indices2]

        # Remove indices that have already been assigned to a ground truth peak
        detected_indices = remove_elements(detected_indices, assigned)



        # If no peaks are detected, continue to the next ground truth peak
        if detected_indices.numel() == 0:
                continue
        else:
            
            dist2 = torch.abs(spec_pk_supp_angles[detected_indices] - angles[k])
            dist2 = torch.min(dist2, angular_grid_fov - dist2)
            closest_index = torch.argsort(dist2)[0]
            closest_index = detected_indices[closest_index]

            assigned = torch.cat((assigned, torch.tensor([closest_index])))

            detections[k] = 1
            if metric == 'rmse':
                #Use closest detected peak to compute the angular error for the kth ground truth peak
                dist = torch.abs(spec_pk_supp_angles[closest_index] - angles[k])
                dist = torch.min(dist, angular_grid_fov - dist)
                errors[k] = dist ** 2
        

    if metric == 'detection_rate':
        return torch.count_nonzero(detections) / len(detections)
    
    elif metric == 'rmse':
        if torch.sum(detections) != 0:
            return torch.sum(errors) / torch.sum(detections)
        else:
            return float('nan')
           
    elif metric in ['false_alarm_rate']:
        # 1. Mask out peaks that were assigned to true targets
        mask = torch.ones(len(spec_pk_amp), dtype=torch.bool, device=spectrum.device)
        if assigned.numel() > 0:
            mask[assigned.long()] = False

        # 2. Extract amplitudes of unassigned peaks
        unassigned_peak_amps = spec_pk_amp[mask]

        # 3. Count unassigned peaks exceeding the threshold (False Alarms)
        false_alarm_count = torch.sum(unassigned_peak_amps > false_alarm_threshold).item()

        # 4. Total possible false peak slots in the candidate pool
        max_candidate_peaks = math.ceil(len(spectrum) / 2)
        num_true_targets = len(angles)
        eligible_null_peak_slots = max_candidate_peaks - num_true_targets

        if eligible_null_peak_slots <= 0:
            return float('nan')

        return false_alarm_count / eligible_null_peak_slots






def average_metric(results, num_vectors_per_variance=1e3, metric='detection_rate'):
    """
    Returns the average metric over a batch of size 'num_vectors_per_variance' for each variance value in the test dataset. The input 'results' is a tensor of shape (num_test_vectors_per_variance * num_variance_values,) containing the computed metric values for each test vector in the test dataset in contiguous blocks of length 'num_vectors_per_variance' corresponding to each variance value.

    Inputs:
        results (torch.Tensor): The computed metric values for each test vector in the test dataset of shape (num_test_vectors_per_variance * num_variance_values,).
        num_vectors_per_variance (int): The number of test vectors corresponding to each variance value in the test dataset.
        metric (str): The metric to compute. Options are 'detection_rate', 'rmse', 'false_alarm_rate', 'reconstruction_contrast'.
    Outputs:
        average (torch.Tensor): The average metric value for each variance value in the test dataset of shape (num_variance_values,).
    """
    num_variance_values = results.shape[0] // num_vectors_per_variance
    average = torch.zeros(num_variance_values, dtype=torch.float64)

    for n in range(num_variance_values):

            batch_variance = results[int(n * num_vectors_per_variance):int((n + 1) * num_vectors_per_variance)]
            batch_variance = batch_variance[~torch.isnan(batch_variance)]

            if batch_variance.numel() == 0:
                #if no detections were made for this batch, return -100 for the average metric value
                average[n] = float('-100')
                continue
            else:
                average[n] = torch.sum(batch_variance) / len(batch_variance)
                if metric == "rmse":
                    average[n] = torch.sqrt(average[n])

    return average.tolist()






def evaluate_framework(framework, dataset_tag, device='cpu', **kwargs):
    """
    Evaluates the performance of a sparse reconstruction framework on a given metric using a test dataset and saves the results in the outputs/metrics and outputs/spectrums directories.

    Inputs:
        framework (str): The name of the framework to evaluate. Options are 'LASSO', 'L1L2', 'TL1', 'MCP','SR-LASSO', 'SR-TL1', 'SBL'.
        dataset_tag (str): The tag of the test dataset. This can be generated with the dataset_generator function in the dataset folder.
        device (str): The device to use for computation. Options are 'cpu' or 'cuda'.
        **kwargs: Additional keyword arguments specifying the framework configuration and metric parameters. See the FRAMEWORK CANONICAL_SIGNATURES dictionary in frameworks/frameworks_wrapper.py for the expected parameters for each framework. If no metric keyword argument is provided, the function will only compute and save the spectrums without evaluating any metrics. If the metric is specified, the following additional keyword arguments are required:
            metric (str): The metric to evaluate. Options are 'detection_rate', 'rmse', 'false_alarm_rate', 'false_discovery_rate'.
            angular_bins_threshold (int): An estimated peak passes the first stage of detection if it is within angular_bins_threshold * angular_grid_stepsize of a ground truth peak's angular position.
            amplitude_threshold (float): An estimated peak passes the second stage of detection if its amplitude is greater than amplitude_threshold * ground_truth_amplitude. If an estimated peak passes both stages of detection, it is considered a true positive detection. If the estimated peak fails either stage of detection, it is considered a false positive detection.
            false_alarm_threshold (float): The threshold for false alarm rate metric. Required only for 'false_alarm_rate' metric. A false positive detection is registered as a false alarm and contributes to the false alarm rate if its amplitude is greater than false_alarm_threshold.
    
    If the spectrum doesn't not already exist for the specific dataset and framework configuration, it will be computed and saved in the outputs/spectrums/{dataset_tag}/{framework_name}/{framework_config_tag} directory with a unique 8-character alphanumeric tag resulting from the specific framework configuration and dataset used. It will additionally be logged in the manifest_spectrums.json and paths_spectrums.json files under with that same tag If the metric is specified, the metric will be computed and saved in the outputs/metrics directory with a unique 8-character alphanumeric tag. 

    If the metric result from a specific dataset, framework configuration, and metric configuration doesn't already exist, the function will evaluate the metric, create a unique 8-character alphanumeric tag for the specific combination of dataset, framework configuration, and metric configuration, and save the results in the outputs/metrics/{dataset_tag}/{framework_name}/{metric_name}/{framework_config_tag} directory. It will additionally be logged in the manifest_metrics.json and paths_metrics.json files under with that same tag. If the metric result already exists, it will be loaded from the outputs/metrics directory and returned.

    The function uses caching to avoid redundant computations. If the spectrum for a specific dataset and framework configuration has already been computed, it will be loaded from the cache instead of being recomputed. The cache is stored in the evaluate_framework._cache attribute, which is a dictionary that maps spectrum tags to their corresponding spectrums.
    """

    
    dataset_metadata = key_management(MANIFEST_DATASETS, dataset_tag, mode='load')
    dataset_path = key_management(PATHS_DATASETS, dataset_tag, mode='load')
    
    framework_label = framework.replace("-", "").lower()

    for path in [PATHS_SPECTRUMS, MANIFEST_SPECTRUMS, PATHS_METRICS, MANIFEST_METRICS]:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{}", encoding="utf-8")

    spectrum_state, spectrum_setup, framework_configuration, spectrum_tag, framework_tag = spectrum_exists(framework, dataset_tag, **kwargs)
    
    spectrum_path = f"{OUTPUTS_PATH}/spectrums/{dataset_tag}/{framework_label}/{framework_tag}"


    # check all the correct metric parameters are provided if metric is specified
    if "metric" in kwargs:

        metric_configuration = {}
        metric = metric_configuration['metric'] = kwargs.pop('metric')

        if metric not in ['detection_rate', 'rmse', 'false_alarm_rate']:
            raise ValueError(f"Invalid metric '{metric}'. Supported metrics are: 'detection_rate', 'rmse', 'false_alarm_rate'.")

        thresholds = ["angular_bins_threshold", "amplitude_threshold"]

        if metric in ['false_alarm_rate']:
            thresholds.append("false_alarm_threshold")

        missing_required = [name for name in thresholds if name not in kwargs]
        if missing_required:
            raise ValueError(f"Missing required arguments for metric {metric}: {missing_required}")


        amplitude_threshold = metric_configuration['amplitude_threshold'] = kwargs.pop('amplitude_threshold')
        angular_bins_threshold = metric_configuration['angular_bins_threshold'] = kwargs.pop('angular_bins_threshold')
        

        threshold_values = [amplitude_threshold, angular_bins_threshold]
        if  metric in ['false_alarm_rate']:
            false_alarm_threshold = metric_configuration['false_alarm_threshold'] = kwargs.pop('false_alarm_threshold')
            threshold_values.append(false_alarm_threshold)
        else:
            false_alarm_threshold = None
            kwargs.pop('false_alarm_threshold', None)  # Remove if present, else ignore

        metric_name = metric.replace('_', ' ')
    else:
        print(f"No metric specified. Only computing and saving the spectrum for framework '{framework}' with configuration {framework_configuration}. If you want to evaluate a (metric, value) kwarg, please specify the metric and its parameters in the function call. Supported metric values are: 'detection_rate', 'rmse', 'false_alarm_rate'.")

    #check if the spectrum for that dataset and framework configuration already exists, if not compute it and save it in the outputs/spectrums directory.
    if spectrum_state:
        print(f"Spectrum results for framework '{framework}' with configuration {framework_configuration} already exist under tag '{spectrum_tag}' in {spectrum_path}.")                
    else:
        print(f"Computing spectrums for framework '{framework}' with configuration {framework_configuration}...")  

        dataset = torch.load(dataset_path, weights_only=True)

        dictionary_tag = dataset_metadata['dictionary_tag']
        dictionary_path = key_management(PATHS_DICTIONARIES, dictionary_tag, mode='load')
        dictionary = torch.load(dictionary_path, weights_only=True)['dictionary']

        measurement_vectors = dataset['data']['measurement_vectors'].T

        spectrums = wrapper(framework, device=device, dictionary=dictionary, measurement_vectors=measurement_vectors, **kwargs).T

        spectrums_dict = spectrum_setup | {'spectrums': spectrums} 

        os.makedirs(spectrum_path, exist_ok=True)
        spectrum_file = f"{spectrum_path}/{spectrum_tag}.pt"

        framework_state = key_management(MANIFEST_FRAMEWORKS, framework_tag, mode='query')
        if not framework_state:
            key_management(MANIFEST_FRAMEWORKS, framework_tag, mode='save', object=framework_configuration)

        torch.save(spectrums_dict, spectrum_file)
        key_management(MANIFEST_SPECTRUMS, spectrum_tag, mode='save', object=spectrum_setup)
        key_management(PATHS_SPECTRUMS, spectrum_tag, mode='save', object=spectrum_file)

        print(f"Spectrum results saved with tag '{spectrum_tag}' under {spectrum_path}.")

    
    #If no metric is specified, return after computing the spectrum
    if not metric_configuration:
        return

    metric_configuration['dataset_tag'] = dataset_tag
    
    metric_tag = generate_8char_tag(metric_configuration | {"spectrum_tag": spectrum_tag})
    metric_state = key_management(MANIFEST_METRICS, metric_tag, mode='query')
    metric_path = f"{OUTPUTS_PATH}/metrics/{dataset_tag}/{framework_label}/{metric}/{framework_tag}"

    if metric_state:
        print(f"Metric results for dataset '{dataset_tag}', framework '{framework}' with configuration {framework_configuration}, and metric '{metric_name}' with configuration {metric_configuration} already exist under tag '{metric_tag}' in {os.path.dirname(metric_path)}. ")
        return
    
    else:
        #If multiple metric compute calls are made for the same dataset and framework configuration, we can cache the spectrum to avoid recomputing it.
        _cache = evaluate_framework._cache

        if _cache is not None and f"{spectrum_tag}" in _cache:
            spectrums = _cache[f"{spectrum_tag}"]
        else:
            if spectrum_state:
                spectrum_file = key_management(PATHS_SPECTRUMS, spectrum_tag, mode='load')
                spectrums  = torch.load(spectrum_file, weights_only=True)['spectrums']
        
        evaluate_framework._cache = {f"{spectrum_tag}": spectrums}

        dataset = torch.load(dataset_path, weights_only=True) 
        angles = dataset['data']['angles']
        amplitudes = dataset['data']['amplitudes']
        dictionary_length = dataset_metadata['dictionary_length']

        angular_grid = torch.arange(-90, 90, 180/dictionary_length, dtype=torch.float64)
        num_test_vectors = angles.shape[0]

        results = torch.zeros(num_test_vectors, dtype=torch.float64)

        
        for s in range(num_test_vectors):

            results[s] = metric_estimate(metric,
                torch.abs(spectrums[s, :]),
                angular_grid,
                angles[s, :],
                amplitudes[s, :],
                angular_bins_threshold,
                amplitude_threshold,
                false_alarm_threshold
            )
            progress = (s + 1) / num_test_vectors * 100

            print(f"\rComputing {metric_name} with ({', '.join(thresholds)}) = ({', '.join(map(str, threshold_values))}): {progress:.2f} %", end='', flush=True)
                        

        average = average_metric(results, dataset_metadata['num_vectors_per_variance'], metric)

        metric_dict = metric_configuration | framework_configuration | {'average': average}
        metric_dict['metric_tag'] = metric_tag

        metric_metadata = metric_configuration | {"dataset_tag": dataset_tag, "framework": framework, "framework_configuration": framework_tag, "spectrum_tag": spectrum_tag, "average": average}
        metric_dict['metric_tag'] = metric_tag

            
        print(f"Metric results for framework '{framework}' with configuration {framework_configuration} and metric '{metric_name}' saved under tag '{metric_tag}' in {os.path.dirname(metric_path)}.")
        
        os.makedirs(metric_path, exist_ok=True)
        metric_file = f"{metric_path}/{metric_tag}.pt"

        torch.save(metric_dict, metric_file)
        key_management(MANIFEST_METRICS, metric_tag, mode='save', object=metric_metadata)
        key_management(PATHS_METRICS, metric_tag, mode='save', object=metric_file)

evaluate_framework._cache = None
