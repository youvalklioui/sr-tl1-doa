import torch


from frameworks.operators import prox_l1_update, subgradient
from frameworks.sbl import sbl




SHARED = ['measurement_vectors', 'dictionary', 'regularization_parameter', 'iterations']

FRAMEWORKS_CANONICAL_SIGNATURES = {
    'LASSO': SHARED + ['rho'],
    'SR-LASSO': SHARED + ['rho1', 'rho2'],
    'TL1': SHARED + ['outer_iterations', 'rho', 'alpha'],
    'MCP': SHARED + ['outer_iterations', 'rho', 'gamma'],
    'L1L2': SHARED + ['outer_iterations', 'rho'],
    'SR-TL1': SHARED + ['outer_iterations', 'rho1', 'rho2', 'alpha'],
    'SBL': ['measurement_vectors', 'dictionary', 'iterations']
}


def wrapper(framework, device='cpu', **kwargs):

    """
    Wrapper function to call the appropriate optimization framework based on the provided framework name.
    
    Inputs:
        framework (str): Name of the optimization framework to use. Supported frameworks are: 'LASSO', 'SR-LASSO', 'TL1', 'MCP', 'L1L2', 'SR-TL1', 'SBL'.
        device (str): Device to run the computations on. Default is 'cpu'. Can be set to 'cuda' for GPU acceleration.
        **kwargs: Additional keyword arguments specific to the chosen framework. These should match the expected parameters for the selected framework.
        The expected parameters for each framework are defined in the FRAMEWORKS_CANONICAL_SIGNATURES dictionary.
        'measurement_vectors' (Tensor): Tensor of shape (M, batch_size) representing the observed measurements.
        'dictionary' (Tensor): Tensor of shape (M, N) representing the measurement matrix.
        'regularization_parameter' (float): Regularization parameter for the optimization problem (kappa).
        'iterations' (int): Number of iterations for the ADMM algorithm.
        'outer_iterations' (int): Number of outer iterations for DCA-based frameworks (TL1, MCP, L1L2, SR-TL1).
        'rho' (float): ADMM penalty parameter for frameworks that use it (LASSO, TL1, MCP, L1L2).
        'rho1' (float): ADMM penalty parameter for the first constraint in SR-LASSO and SR-TL1 frameworks.
        'rho2' (float): ADMM penalty parameter for the second constraint in SR-LASSO and SR-TL1 frameworks.
        'alpha' (float): Hyperparameter for TL1 and SR-TL1 frameworks.
        'gamma' (float): Hyperparameter for MCP framework. See the MCP subgradient in frameworks/operators.py for more details on the MCP implementation.
    Outputs:
        Tensor of shape (N, batch_size) representing the reconstructed signal.
    """

    framework_params = FRAMEWORKS_CANONICAL_SIGNATURES.get(framework)

    if framework_params is None:
        raise ValueError(f"Unknown framework: {framework}. Supported frameworks are: {list(FRAMEWORKS_CANONICAL_SIGNATURES.keys())}")

    missing_required = [
        name for name in framework_params
        if name not in kwargs
    ]
    
    # 2. Identify unexpected keyword arguments
    unexpected_kwargs = [k for k in kwargs if k not in framework_params]
    
    # If anything is wrong, raise an informative error
    if missing_required or unexpected_kwargs:
        error_msgs = []
        if missing_required:
            error_msgs.append(f"Missing required arguments for {framework}: {missing_required}")
        if unexpected_kwargs:
            error_msgs.append(f"Unexpected arguments provided for {framework}: {unexpected_kwargs}")
            
        expected_sig = [name for name in framework_params]
        error_msgs.append(f"Expected signature parameters for {framework}: {expected_sig}")
        
        raise TypeError("\n".join(error_msgs))


    y = kwargs.get('measurement_vectors').to(device) 
    D = kwargs.get('dictionary').to(device)  

    # Normalize the columns of the dictionary to have unit norm.
    R = 1 / torch.norm(D, dim=0)
    D = D * R.unsqueeze(0)

    regularization_parameter = kwargs.get('regularization_parameter')


    if framework in ['TL1', 'SR-TL1']:
        hyperparameter = kwargs.get('alpha')
    elif framework in ['MCP']:
        hyperparameter = kwargs.get('gamma')
    else:
        hyperparameter = None
    

    # 'iterations' for DCA-based frameworks (TL1, MCP, L1L2, SR-TL1) corresponds to the number of iterations for ADMM for solving the DCA subproblem.  'outer_iterations' corresponds to the number of DCA iterations. For other frameworks, 'iterations' simply corresponds to the total number of iterations, also solved using ADMM.
    iterations = kwargs.get('iterations')
    outer_iterations = kwargs.get('outer_iterations') if framework in ['L1L2','TL1', 'MCP', 'SR-TL1'] else 1

    total_iterations = iterations * outer_iterations



    N = D.shape[1]

    #Compact SVD of D: D = U Sigma V^H, we only need the singular values and V for the efficient ADMM updates.
    _, Sigma, V = torch.linalg.svd(D, full_matrices=False)

    
    
    x = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)
    

    if framework in ['LASSO', 'TL1', 'L1L2', 'MCP']:

        
        rho = kwargs.get('rho')

        H = 1/rho - 1 / (Sigma**2 + rho)
        V = V.conj().T * torch.sqrt(H)

        u = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)
        z = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)

        yf = D.T.conj() @ y

        for l in range(total_iterations):

            s = rho * (z - u) + yf

            if framework != 'LASSO':
                #The subgradient is updated every outer_iteration since it is constant for a given outer_iteration.
                if l % iterations == 0:
                    subgrad = subgradient(framework, x, hyperparameter)
                    
                s += 2 * regularization_parameter * subgrad

            x = -V @ (V.T.conj() @ s) + s/rho

            z = prox_l1_update(framework, x + u, regularization_parameter, rho, hyperparameter=hyperparameter)

            u = u + x - z
        


    elif framework in ['SR-LASSO', 'SR-TL1']:

        rho1 = kwargs.get('rho1')
        rho2 = kwargs.get('rho2')

        if framework in ['SR-TL1']:
            rho2_tilde = rho2 * (1 + 1e-8)
            H = 1/rho2_tilde - 1 / (rho1 * Sigma**2 + rho2_tilde)
        else:
            H = 1/rho2 - 1 / (rho1 * Sigma**2 + rho2)

        V = V.conj().T * torch.sqrt(H)

        #ADMM auxilary and dual variables are initialized to zero.  The auxilary variables are z1 and z2, corresponding to the constraints (z1 = D @ x and z2 = x) and the dual variables are u1 and u2.
        u1 = torch.zeros((D.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
        u2 = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)
        z1 = torch.zeros((D.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
        z2 = torch.zeros((N, y.shape[1]), dtype=y.dtype, device=y.device)

        for l in range(total_iterations):
            
            s = rho1 * D.conj().T @ (z1-u1) + rho2*(z2-u2) 

            if framework != 'SR-LASSO':
                #The subgradient is updated every outer_iteration since it is constant for a given outer_iteration.
                if l % iterations == 0:  
                    subgrad = subgradient(framework, x, hyperparameter)

            
                s = s + 2 * regularization_parameter * subgrad

            if framework == 'SR-TL1':
                x = -V @ (V.T.conj() @ s) + s/rho2_tilde
            else:
                x = -V @ (V.T.conj() @ s) + s/rho2

            xf = D @ x

            q = xf + u1 - y

            z1 = y + torch.max(1-1/(rho1*torch.norm(q, dim=0)),torch.zeros_like(torch.norm(q, dim=0)))*q

            z2 = prox_l1_update(framework, x + u2, regularization_parameter, rho2=rho2, hyperparameter=hyperparameter)

            u1 = u1 + xf - z1 
            u2 = u2 + x - z2

    elif framework == 'SBL':
        x = sbl(measurement_vectors=y, dictionary=D, iterations = total_iterations)

    #Rescaling the solution to account for the normalization of the dictionary columns.
    r = R.unsqueeze(-1) * x
    r = r.to('cpu')

    return r


