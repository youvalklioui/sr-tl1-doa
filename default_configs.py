DEFAULT_FRAMEWORKS_CONFIGS = {
    'LASSO': {
        'regularization_parameter': 0.3,
        'iterations': 1000,
        'rho': 1.0},

    'L1L2': {
        'regularization_parameter': 0.3,
        'iterations': 20,
        'outer_iterations': 50,
        'rho': 1.0},

    'TL1': {
        'regularization_parameter': 0.15,
        'iterations': 20,
        'outer_iterations': 50,
        'rho': 1.0,
        'alpha': 1.0},

    'MCP': {
        'regularization_parameter': 0.2,
        'iterations': 20,
        'outer_iterations': 50,
        'rho': 1.0,
        'gamma': 0.5},

    'SR-LASSO': {
        'regularization_parameter': 0.3,
        'iterations': 1000,
        'rho1': 1.0,
        'rho2': 10.0},

    'SR-TL1': {
        'regularization_parameter': 0.2,
        'iterations': 20,
        'outer_iterations': 50,
        'rho1': 1.0,
        'rho2': 10.0,
        'alpha': 1.0},

    'SBL': {
        'iterations': 1000}
    
}


DEFAULT_METRICS_CONFIGS = {
    "amplitude_threshold": 0.1,
    "angular_bins_threshold": 2,
    "false_alarm_threshold": 0.005,
}


DEFAULT_FRAMEWORKS_REGULARIZERS = {
    'LASSO': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
    'L1L2': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,  0.7],
    'TL1': [0.01, 0.05, 0.1, 0.15,  0.2, 0.3, 0.4,  0.5,  0.7],
    'MCP': [0.01, 0.05, 0.1, 0.2, 0.5,  0.7, 1, 1.3, 1.5],
    'SR-LASSO':[0.01, 0.05, 0.1, 0.15, 0.25, 0.27, 0.29, 0.3, 0.4],
    'SR-TL1': [0.01, 0.05, 0.1, 0.15, 0.17, 0.19, 0.2, 0.25, 0.3],
    'SBL': [None]
}


SETTINGS = {
    "create-array": {
        "num_elements": 30,
        "aperture": 60
    },
    "create-dictionary":{
        "dictionary_length":256,
        "max_gain_deviation": 0.3,
        "max_phase_deviation": 30.0,
        "correlation_length": 5,
        "average_mutual_coupling": 0.1,
        "relative_variation_coupling": 0.1
    },
    "create-dataset": {
        "log_noise_variance_values": [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0],
        "num_vectors_per_variance": 1000,
        "number_sources": 10,
        "min_freq_separation_factor": 3
    }}