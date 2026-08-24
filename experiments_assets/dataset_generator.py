import os
import json
from pathlib import Path
import torch

from utils.assets_management_utils import generate_8char_tag, key_management

from utils.paths import MANIFEST_DICTIONARIES, PATHS_DICTIONARIES, MANIFEST_DATASETS, PATHS_DATASETS, MANIFEST_ARRAYS


PI = torch.tensor(torch.pi, dtype=torch.float64)



def randu(shape, a, b):
    """Generates a tensor of dimensions `shape` with samples from a uniform distribution over [a, b].

    Inputs:
        shape (tuple): The desired shape of the output tensor.
        a (float): The lower bound of the uniform distribution.
        b (float): The upper bound of the uniform distribution.
    Outputs:
        torch.Tensor: A tensor of the specified shape with samples drawn from the uniform distribution over [a, b]."""
    
    a = torch.tensor(a, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)
    random_tensor = a + (b - a) * torch.rand(shape, dtype = torch.float64)
    return random_tensor



def randn(shape, mean, std_dev):
    """Generates a tensor of dimensions `shape` with samples from a Normal distribution N(mean, std_dev).
    Inputs:
        shape (tuple): The desired shape of the output tensor.
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.
    Outputs:
        torch.Tensor: A tensor of the specified shape with samples drawn from the normal distribution N(mean, std_dev)."""

    random_tensor = mean + std_dev * torch.randn(shape)
    return random_tensor




def randi(low, high, k):
    """
    Outputs a tensor of k integers drawn without replacement from the interval [low, high] (inclusive).

    Inputs:
        low (int): Lower bound of the interval.
        high (int): Upper bound of the interval.
        k (int): Number of integers to draw.
    Outputs:
        torch.Tensor: A tensor of k unique integers drawn from the specified interval.
    """
    # Ensure the size is not larger than the number of possible values
    assert k <= (high - low + 1), "k cannot be larger than the number of available integers."

    all_integers = torch.arange(low, high + 1)
    random_selection = all_integers[torch.randperm(all_integers.size(0))][:k]

    return random_selection



def rand_freqs(f1, f2, df, k):
    """
    Generate k random frequencies in [f1, f2] with minimum separation df.
    Inputs:
        f1 (float): Lower bound of the frequency interval.
        f2 (float): Upper bound of the frequency interval.
        df (float): Minimum separation between frequencies.
    Outputs:
        torch.Tensor: A tensor of k frequencies satisfying the constraints. 
    """


    # Check if the interval can hold k points with the required spacing
    if (f2 - f1) < (k - 1) * df:
        raise ValueError(
            f"Interval [{f1}, {f2}] is too small for {k} frequencies "
            f"with minimum distance {df}."
        )

    # Draw k uniform samples in the reduced interval [0, (f2-f1) - (k-1)*df],
    # sort them, then re-introduce the mandatory gaps i*df between neighbours.
    # This gives a uniformly random placement over the whole interval.
    reduced = (f2 - f1) - (k - 1) * df
    samples, _ = torch.sort(torch.rand(k, dtype=torch.float64) * reduced)
    freqs = f1 + samples + torch.arange(k, dtype=torch.float64) * df

    return freqs



def generate_array(num_elements, aperture, custom_array_indices=None):
    """
    Generates a random sparse linear array_indices (SLA) given an aperture and number of elements. A unique 8-character alphanumeric ID for the array configuration is created and the array topology (linear indices in λ/2 units) is saved in the manifest_arrays.json under the arrays directory if it doesn't already exist.

    Inputs:
        num_elements (int): Number of elements in the array.
        aperture (int): Aperture of the array in λ/2 units.
        custom_array_indices (list, optional): Custom array_indices positions. If provided, num_elements and aperture are ignored.
    Outputs:
        array_tag (str): An 8-character alphanumeric ID for the generated array configuration.
    """

    if custom_array_indices is not None:
        if not all(type(x) is int for x in custom_array_indices):
            raise ValueError("custom_array_indices must be a list of integers.")
        array_indices = custom_array_indices
    else:
        if not num_elements or not aperture:
            raise ValueError("Both num_elements and aperture must be provided if custom_array_indices is not given.")
        else:
            is_valid_topology = num_elements - 1 <= aperture
            if not is_valid_topology:
                raise ValueError("Invalid array topology: num_elements - 1 must be less than or equal to aperture.")
        # Generate and sort random sensor positions between 1 and (aperture - 1)
        array_indices, _ = torch.sort(randi(1, aperture - 1, num_elements - 2))
        # Append fixed endpoints 0 and aperture.
        array_indices = torch.cat((torch.tensor([0]), array_indices, torch.tensor([aperture]))).tolist()

    if not os.path.exists(MANIFEST_ARRAYS):
        os.makedirs(os.path.dirname(MANIFEST_ARRAYS), exist_ok=True)
        with open(MANIFEST_ARRAYS, "w", encoding="utf-8") as f:
            json.dump({}, f)
            
    array_tag = generate_8char_tag({"array_indices": array_indices})
    array_state = key_management(MANIFEST_ARRAYS, array_tag, mode = "query")
    
    if not array_state:
        key_management(MANIFEST_ARRAYS, array_tag, mode = "save", object={"aperture": aperture, "num_elements": num_elements, "array_indices": array_indices})

    status = "already exists" if array_state else "saved"
    
    print(f"Array with {num_elements} elements and aperture {aperture}λ/2 {status} under key {array_tag} in {MANIFEST_ARRAYS}.")

    return array_tag
    



def matern32_kernel(angular_grid, correlation_length=10): 
    """Computes a correlation matrix based on a Matern 3/2 kernel to be used for generating angular-dependent amplitude and phase mismatch patterns.

    Inputs:
        angular_grid (torch.Tensor): A 1D tensor of angles of shape (N,) representing the angular grid.
        correlation_length (float): The correlation length parameter for the Matern kernel in degrees. Default is 10 degrees. A larger correlation length results in smoother mismatch patterns, while a smaller correlation length results in more rapid variations in the mismatch pattern.
    Outputs:
        torch.Tensor: A 2D tensor of shape (N, N) representing the correlation matrix based on the Matern 3/2 kernel.
    """

    d = torch.abs(angular_grid[:, None] - angular_grid[None, :])

    kernel = (1 + 3**0.5*d/correlation_length) * torch.exp(-3**0.5*d/correlation_length)

    return kernel


def create_standard_mismatch_pattern(kernel):
    """Creates a zero mean Gaussian mismatch pattern using a given covariance kernel. The standard mismatch pattern is used to generate angular-dependent amplitude and phase mismatch patterns.
    
    Inputs:
        kernel (torch.Tensor): A 2D tensor of shape (N, N) representing the covariance matrix for the Gaussian distribution.
    Outputs:
        torch.Tensor: A 1D tensor of shape (N,) representing the generated zero-mean standard mismatch pattern.
    """

    mean = torch.zeros(kernel.size(0), dtype=kernel.dtype, device=kernel.device)

    multivariate_gaussian_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=kernel)

    standard_mismatch_pattern = multivariate_gaussian_dist.sample()

    return standard_mismatch_pattern



def generate_mismatch_pattern(angular_grid, correlation_length=10, max_mismatch_deviation=0.3, mismatch_type='gain'):
    """Creates an angular-dependent gain or phase mismatch pattern for a given element of the antenna array based on a Gaussian process and a Matern 3/2 kernel. The correlation length is used to build the Matern 3/2 kernel and determines how quickly the mismatch pattern varies over the angular grid. The Gaussian mismatch pattern is then scaled to have a maximum deviation with respect to its mean of max_mismatch_deviation. The mean is equal to 1 for gain mismatch and 0 for phase mismatch.
    
    Inputs:
        angular_grid (torch.Tensor): A 1D tensor of angles of shape (N,) representing the angular grid.
        correlation_length (float): The correlation length parameter for the Matern 3/2 kernel in degrees.
        max_mismatch_deviation (float): The maximum deviation of the mismatch pattern from its mean. For gain mismatch, the mean is 1, and for phase mismatch, the mean is 0. The mismatch pattern will vary exactly in [mean - max_mismatch_deviation, mean + max_mismatch_deviation].
        mismatch_type (str): Type of mismatch pattern to generate. Can be 'gain' or 'phase'.
    Outputs:
        torch.Tensor: A 1D tensor of shape (N,) representing the generated angular-dependent mismatch pattern.
    """

    #Create the Matern 3/2 kernel and generate a standard mismatch pattern
    kernel = matern32_kernel(angular_grid, correlation_length=correlation_length)
    scaled_mismatch_pattern = create_standard_mismatch_pattern(kernel)

    #Scale the mismatch pattern to have a maximum deviation of max_mismatch_deviation with respect to its mean
    range = torch.max(scaled_mismatch_pattern) - torch.min(scaled_mismatch_pattern)
    scaling_factor = 2 * max_mismatch_deviation / range
    scaled_mismatch_pattern = scaling_factor * scaled_mismatch_pattern

    #Translate the mismatch pattern to have the correct mean (1 for gain, 0 for phase) based on the mismatch type
    if mismatch_type == 'gain':
        translation = (1 + max_mismatch_deviation) - torch.max(scaled_mismatch_pattern)
    elif mismatch_type == 'phase':
        translation = max_mismatch_deviation - torch.max(scaled_mismatch_pattern)

    scaled_mismatch_pattern = scaled_mismatch_pattern + translation
    
    return scaled_mismatch_pattern



def generate_mutual_coupling_matrix(array_indices, average_mutual_coupling=0.1, relative_variation_coupling=0.1):

    """Generates a complex-valued mutual coupling matrix Γ of shape (M,M) for an M-element antenna array based on the average mutual coupling, relative variation in coupling, and the indices of the array elements. 
    
    Inputs:
        array_indices (torch.Tensor): A 1D tensor of shape (M,) representing the integer indices of the antenna array elements specified. Here we assume the elements lie on a λ/2 grid.
        average_mutual_coupling (float): The average mutual coupling value between two array elements with a distance of λ/2 apart.  Array elements that are further apart will have a coupling coefficient that is a fraction of this value with the fraction being inversely proportional to the distance between the two elements.  

        relative_variation_coupling (float): This parameter controls the standard deviation of the mutual coupling amplitude around its mean. The higher this value the higher the variation in mutual coupling there will be for the same distance between two array elements.
    Outputs:
        torch.Tensor: A 2D tensor of shape (M, M) representing the generated mutual coupling matrix for the antenna array. The diagonal elements are 1 (self-coupling), and the off-diagonal elements represent the mutual coupling between adjacent elements, with random variations based on the specified average and relative variation.
    
    """

    # the coupling coefficient amplitude between array element m and m + 1 is A1n/|k_m+1-k_m| (1 + sqrt(3) * A2 * randu([-1,1])) where k_m is the linear index of the m-th array element on the λ/2 grid. The phase of the coupling coefficient is random in [0, 2π]. See section 6.2.1 in manuscript for more details.

    array_size = len(array_indices)


    Gamma = torch.eye(array_size, dtype=torch.cdouble)

    # We consider only the coupling between adjacent elements, i.e. the immediate neighbors.
    distances = torch.abs(array_indices - torch.roll(array_indices, shifts=-1))[:-1]


    coupling = average_mutual_coupling / distances * (1 + 3**0.5 * relative_variation_coupling 
                                                      * randu(distances.shape, -1,1))

    coupling = coupling * torch.exp(1j * 2 * PI * randu(coupling.shape, 0,1))

    Gamma.diagonal(-1).copy_(coupling.squeeze())
    Gamma.diagonal(1).copy_(coupling.squeeze())

    return Gamma



def generate_dictionary(array_tag, dictionary_length=256, correlation_length=5, max_gain_deviation=0.3,    
                        max_phase_deviation=30.0, average_mutual_coupling=0.1, relative_variation_coupling=0.1):
    """
    Generate a dictionary A of shape (M, N) where M is the number of elements in the array and N is the length of the angular grid (which spans from [-90, 90] with stepsize 180 /dictionary_length). A = Γ @ (Ψ * A_ideal) where Γ is the (M,M) mutual coupling matrix and Ψ is the (M,N) complex gain-phase mismatch matrix and A_ideal is the ideal array manifold. See the Signal Model section and section 6.2.1 in the manuscript for additional details on the imperfections modeling. A unique 8-character alphanumeric ID is generated for the dictionary based on its parameters and is logged in both the manifest_dictionaries.json and paths_dictionaries.json in the dictionaries directory. The dictionary itself is saved in a .pt file under dictionaries directory with a filename corresponding to the generated dictionary ID. 

    Inputs:
        array_tag (str): The unique identifier for the array configuration, which can be found in the MANIFEST_ARRAYS file.
        dictionary_length (int): The length N of the angular grid, which determines the number of columns in the dictionary.
        correlation_length (int): The correlation length for the gain-phase mismatch patterns. Higher values result in smoother mismatch patterns across the angular grid.
        max_gain_deviation (float): The maximum gain deviation for the gain-phase mismatch patterns.
        max_phase_deviation (float): The maximum phase deviation (in degrees) for the gain-phase mismatch patterns.
        average_mutual_coupling (float): The average mutual coupling coefficient between array elements.
        relative_variation_coupling (float): The relative variation in the mutual coupling coefficients across the array elements.  
    
    Outputs:
        dictionary_tag (str): An 8-character alphanumeric ID for the generated dictionary configuration.

    """
    # Create uniform angular grid from -90 to 90 degrees.

    
    with open(MANIFEST_ARRAYS, "r", encoding="utf-8") as manifest:
          arrays_manifest = json.load(manifest)
    array_config = arrays_manifest.get(array_tag)  
    
    num_elements = array_config['num_elements']
    aperture = array_config['aperture']

    for path in [PATHS_DICTIONARIES, MANIFEST_DICTIONARIES]:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{}", encoding="utf-8")

    

    metadata = {
        'num_elements': num_elements,
        'aperture': aperture,
        'array_tag': array_tag,
        'dictionary_length': dictionary_length,
        'correlation_length': correlation_length,
        'max_gain_deviation': max_gain_deviation,
        'max_phase_deviation': max_phase_deviation,
        'average_mutual_coupling': average_mutual_coupling,
        'relative_variation_coupling': relative_variation_coupling
    }
    dictionary_tag = generate_8char_tag(metadata)
    dictionary_directory = os.path.dirname(MANIFEST_DICTIONARIES)
    dictionary_state = key_management(MANIFEST_DICTIONARIES, dictionary_tag, mode='query')
    
    status = "already exists" if dictionary_state else "saved"

    if not dictionary_state:

        array_indices = torch.tensor(array_config['array_indices'])
        angular_grid = torch.arange(-90, 90, 180 / dictionary_length)

        # Build dictionary matrix using the steering vector formula.
        A_ideal = torch.exp(1j * 2 * PI * array_indices.unsqueeze(-1) * 1/2*torch.sin(torch.deg2rad(angular_grid).unsqueeze(0)))
        Psi = torch.zeros(A_ideal.shape, dtype=A_ideal.dtype)

        # Create a mismatch pattern for each element and stack them to form the mismatch matrix.
        for m in range(len(array_indices)):
            Psi[m, :] = generate_mismatch_pattern(angular_grid, correlation_length=correlation_length, max_mismatch_deviation=max_gain_deviation, mismatch_type='gain') * torch.exp(1j * torch.deg2rad(generate_mismatch_pattern(angular_grid, correlation_length=correlation_length, max_mismatch_deviation=max_phase_deviation, mismatch_type='phase')))

        Gamma = generate_mutual_coupling_matrix(array_indices, average_mutual_coupling, relative_variation_coupling)

        D = Gamma @ (Psi * A_ideal)

        dictionary = {
            'metadata': metadata,
            'dictionary_tag': dictionary_tag,
            'mismatch_matrix': Psi,
            'mutual_coupling_matrix': Gamma,
            'dictionary': D
        }
        
        os.makedirs(dictionary_directory, exist_ok=True)
        dictionary_path = f"{dictionary_directory}/{dictionary_tag}.pt"

        torch.save(dictionary, dictionary_path)
        key_management(MANIFEST_DICTIONARIES, dictionary_tag, mode='save', object=metadata)
        key_management(PATHS_DICTIONARIES, dictionary_tag, mode='save', object=dictionary_path)

    print(f"Dictionary with tag {dictionary_tag} {status} as {dictionary_tag}.pt under {dictionary_directory}.")

    return dictionary_tag




def single_measurement_vector(array_indices, mismatch_matrix, mutual_coupling_matrix, angular_grid, noise_variance, number_sources, min_freq_separation):

    """
    Generates a single measurement vector given array indices (specified in λ/2 units), a mismatch matrix Ψ, a mutual coupling matrix Γ, an angular grid, the noise variance level σ², the number of source K, and minimum normalized frequency separation. The function first generates K random frequencies with a minimum separation, then computes the corresponding angles through the inverse sine, and generates a complex amplitude for each source. Then the measurement vector is built using the ideal steering vectors a(θ), the mismatch matrix Ψ, and the mutual coupling matrix Γ according to eqs. (1), (2), and (3) in the manuscript:
    
    y = Γ @ (Ψ * A_ideal @ s) + n, where A_ideal of shape (M,K) is the ideal array manifold, s of shape (K,1) is the complex amplitude vector, and n ~ N(0, σ² I) is complex Gaussian noise.
    
    The function returns the transposed measurement vector, the angles of the sources, the complex amplitudes, and the SNR of the generated measurement vector.

    Inputs:
        array_indices (torch.Tensor): Sensor array positions specified in λ/2 units, of shape (M,).
        mismatch_matrix (torch.Tensor): Gain-phase mismatch matrix Ψ for the array, of shape (M, N).
        mutual_coupling_matrix (torch.Tensor): Mutual coupling matrix for the array Γ, of shape (M, M).
        angular_grid (torch.Tensor): Angular grid for sparse representation, of shape (N,).
        noise_variance (float): Variance of the additive noise.
        number_sources (int): Number of active sources.
        min_freq_separation (float): Minimum separation between frequencies.

    Outputs:
        tuple: Transposed measurement vector y.T, angles of the sources, complex amplitudes, and SNR of the generated measurement vector.
    """
    
    # Randomly select frequencies with a specified minimum separation.
    freqs = rand_freqs(-1 / 2, 1 / 2 - 1.001/len(angular_grid), min_freq_separation, number_sources)
    angles = torch.rad2deg(torch.asin(2 * freqs))
    
    # Generate complex random amplitudes and phases. The amplitudes are drawn from a uniform distribution in [0.05, 0.3] for half of the sources and [0.7, 1.0] for the other half to have a maximum dynamic range of around 26 dB. The phases are drawn from a uniform distribution in [-π, π].
    amplitudes_small = randu((number_sources//2, 1), 0.05, 0.3).squeeze(-1)
    amplitudes_large = randu((number_sources - number_sources//2, 1), 0.7, 1.0).squeeze(-1)

    amplitudes = torch.cat((amplitudes_small, amplitudes_large), dim=0)

    shuffled_indices = torch.randperm(amplitudes.size(0))
    amplitudes = amplitudes[shuffled_indices]

    phis = randu((number_sources, 1), -PI, PI).squeeze(-1)

    # Create complex amplitudes.
    amplitudes = amplitudes * torch.exp(1j * phis).to(torch.complex128)

    # Build measurement vector using the steering vectors and the mismatch matrix.
    y = torch.zeros((len(array_indices), 1), dtype=torch.complex128)
    for k in range(number_sources):
        idx_mismatch_1 = (torch.abs(angular_grid - angles[k])).argmin()

        if  angles[k] - angular_grid[idx_mismatch_1] > 0:
            idx_mismatch_2 = idx_mismatch_1 + 1
        else:
            idx_mismatch_2 = idx_mismatch_1 - 1

        # Interpolate gain-phase imperfection vector
        psi = mismatch_matrix[:, idx_mismatch_1] + (mismatch_matrix[:, idx_mismatch_2] - mismatch_matrix[:, idx_mismatch_1]) * (angles[k] - angular_grid[idx_mismatch_1]) / (angular_grid[idx_mismatch_2] - angular_grid[idx_mismatch_1])

        # psi = mismatch_matrix[:, idx_mismatch_1]
        psi = psi.unsqueeze(-1)

        a_k = torch.exp(1j * 2 * PI * array_indices * freqs[k]).unsqueeze(-1)

        y +=  (psi * a_k) * amplitudes[k]

    y = mutual_coupling_matrix @ y

    sig_power = 1 / len(y) * torch.sum(torch.abs(y) ** 2)

    # Add complex Gaussian noise.
    sigma = torch.sqrt(torch.tensor([noise_variance]) / 2)
    noise = randn((len(y), 1), 0, sigma) + 1j * randn((len(y), 1), 0, sigma)
    y = y + noise

    snr = 10 * torch.log10(sig_power / noise_variance)

    return y.T, angles, amplitudes, snr




def generate_dataset_test(dictionary_tag, log_noise_variance_values, num_vectors_per_variance, number_sources,
                          min_freq_separation_factor=3):
    """
    Generates a test dataset of measurement vectors for a given dictionary, a list of noise variance levels, the number of vectors per noise variance level, the number of sources, and a minimum frequency separation factor. This function calls the generate_measurement_vector function to create a batch of num_vectors_per_variance for each noise variance value. The normalized frequency separation is calculated as 1 / (min_freq_separation_factor * M), where M is the number of elements in the array. A unique 8-character alphanumeric ID is generated for the dataset based on its parameters and is logged in both the manifest_datasets.json and paths_datasets.json in the dataset directory. The dataset itself is saved in a .pt file under datasets directory with a filename corresponding to the generated dataset ID. 

    Inputs:
        dictionary_tag (str): The unique identifier for the dictionary configuration which can be found in the manifest_dictionaries.json file.
        log_noise_variance_values (list): A list of log noise variance levels for which to generate measurement vectors.
        num_vectors_per_variance (int): The number of measurement vectors to generate for each noise variance level.
        number_sources (int): The number of sources for all the generated measurement vectors.
        min_freq_separation_factor (int, optional): The minimum frequency separation factor. The normalized frequency separation is calculated as 1 / (min_freq_separation_factor * M), where M is the number of elements in the array. 
    
    Outputs:
        dataset_tag (str): An 8-character alphanumeric ID for the generated dataset configuration.
    """

    with open(MANIFEST_DICTIONARIES, "r", encoding="utf-8") as manifest:
          dictionaries_manifest = json.load(manifest)
    with open(MANIFEST_ARRAYS, "r", encoding="utf-8") as manifest:
          arrays_manifest = json.load(manifest)

    dictionary_metadata = dictionaries_manifest.get(dictionary_tag) 
    
    array_metadata = arrays_manifest.get(dictionary_metadata['array_tag']) 
    
    dictionary_path = key_management(PATHS_DICTIONARIES, dictionary_tag, mode='load')

    dataset_directory = os.path.dirname(MANIFEST_DATASETS)

    for path in [PATHS_DATASETS, MANIFEST_DATASETS]:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("{}", encoding="utf-8")
    

    metadata = {
        'num_vectors_per_variance': num_vectors_per_variance,
        'number_sources': number_sources,
        'log_noise_variance_values': log_noise_variance_values,
        'min_freq_separation_factor': min_freq_separation_factor,
        'array_tag': dictionary_metadata['array_tag'],
        'dictionary_tag': dictionary_tag}
    

    metadata = dictionary_metadata | metadata 
    dataset_tag = generate_8char_tag(metadata) 
    dataset_state = key_management(MANIFEST_DATASETS, dataset_tag, mode='query')

    status = "already exists" if dataset_state else "saved"

    if not dataset_state:

        dictionary_length = dictionary_metadata['dictionary_length']

        num_elements = array_metadata['num_elements']
        array_indices = torch.tensor(array_metadata['array_indices'])

        mismatch_matrix = torch.load(dictionary_path, weights_only=True)['mismatch_matrix']
        mutual_coupling_matrix = torch.load(dictionary_path, weights_only=True)['mutual_coupling_matrix']

        angular_grid = torch.arange(-90, 90, 180 / dictionary_length)
        min_freq_separation = 1 / (min_freq_separation_factor * num_elements)
        len_dataset = num_vectors_per_variance * len(log_noise_variance_values)

        measurement_vectors = torch.zeros((len_dataset, num_elements), dtype=torch.complex128)
        angles = torch.zeros((len_dataset, number_sources), dtype=torch.complex128)
        amplitudes = torch.zeros((len_dataset, number_sources), dtype=torch.complex128)
        snrs = torch.zeros((len_dataset, 1), dtype=torch.float64)

        s = 0
        for log_noise_variance in log_noise_variance_values:
            noise_variance = 10 ** log_noise_variance
            for t in range(num_vectors_per_variance):
                measurement_vectors[s, :], angles[s, :], amplitudes[s, :], snrs[s, :] = single_measurement_vector(array_indices, mismatch_matrix, mutual_coupling_matrix, angular_grid, noise_variance, number_sources, min_freq_separation)
                pct = (t + 1) * 100 // num_vectors_per_variance 
                print(f"\rGenerating test dataset for noise variance level = 10^({log_noise_variance}) : {pct} %", end='', flush=True)
                s += 1

        dataset = {
            'metadata': metadata,
            'dataset_tag': dataset_tag,
            'data': {
                'measurement_vectors': measurement_vectors,
                'angles': angles,
                'amplitudes': amplitudes,
                'snrs': snrs
            }
        }
        os.makedirs(dataset_directory, exist_ok=True)
        dataset_path = f"{dataset_directory}/{dataset_tag}.pt"
        torch.save(dataset, dataset_path)
        key_management(MANIFEST_DATASETS, dataset_tag, mode='save', object=metadata)
        key_management(PATHS_DATASETS, dataset_tag, mode='save', object=dataset_path)

    print(f"Dataset with tag {dataset_tag} {status} as {dataset_tag}.pt under {dataset_directory}.")
    return dataset_tag

   