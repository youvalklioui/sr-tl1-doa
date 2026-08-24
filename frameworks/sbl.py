import torch



def sbl(measurement_vectors, dictionary, iterations = 1000):
    """
    Batched Sparse Bayesian Learning (SBL) with per-sample noise variance update.
    
    Args:
        measurement_vectors: A batch of single measurement vectors (SMVs) of shape (M, batch_size).
        iterations: Number of iterations for the SBL algorithm.
        dictionary: Measurement matrix of shape (M, N). Shared across all batches.
    Returns:
        x: Reconstructed signal matrix of shape (N, batch_size).
    """

    y = measurement_vectors
    D = dictionary
    
    M, N = D.shape
    batch_size = y.shape[1]
    device = y.device
    real_dtype = torch.float64

    # Initialize hyperparameters (gamma)
    Gamma = torch.ones((N, batch_size), device=device, dtype=real_dtype)
    
    # Initialize noise variance (sigma) 
    sigma = torch.ones((batch_size,), device=device, dtype=real_dtype) * 1e-2


    DH = D.conj().T
    
    # Pre-compute identity matrix for regularization
    eye_M = torch.eye(M, device=device, dtype=D.dtype)

    eps = 1e-12

    # Pre-expand D and eye_M for batching
    D_exp = D.unsqueeze(0)           # (1, M, N)
    eye_M_exp = eye_M.unsqueeze(0)   # (1, M, M)

    for t in range(iterations):
        

        # Z[k] = D @ diag(gamma_k)
        # Result shape: (batch_size, M, N)
        Z = D_exp * Gamma.t().unsqueeze(1)        
    
        Sigma_y_batch = torch.matmul(Z, DH)
        
        # Add regularization: (sigma + eps) * I
        # sigma: (batch_size,). Reshape to (batch_size, 1, 1) to broadcast to (batch_size, M, M)
        Sigma_y_batch += (sigma + eps).unsqueeze(1).unsqueeze(2) * eye_M_exp

        
        # Prepare y for batch solve: needs to be (batch_size, M)
        y_batch = y.t()                  
        

        # Batch solving for Sigma_y_inv @ y_batch
        Sigma_y_inv = torch.linalg.solve(Sigma_y_batch, eye_M_exp)
        sol_batch = torch.matmul(Sigma_y_inv, y_batch.unsqueeze(-1)).squeeze(-1)

        # Post-multiplication to get x: x = Gamma * D^H @ (Sigma_y_inv @ y)
        x = Gamma * (DH @ sol_batch.t())                   


        # Sigma_ii computation for all batches
        U = torch.matmul(Sigma_y_inv, Z)
        U = torch.real(torch.sum(torch.conj(Z) * U, dim=1)) 
        U = U.t()              
        Sigma_ii = Gamma - U

        # MacKay update rule for gamma: gamma_new = |x|^2 / (1 - Sigma_ii / (Gamma + eps))
        Gamma = torch.abs(x)**2 / torch.clamp((1.0 - (Sigma_ii / (Gamma + eps))), eps)

        # Pruning small values of gamma
        Gamma[Gamma < eps] = 0.0

        # Update noise variance σ²
        Dx_batch = torch.matmul(D_exp, x.t().unsqueeze(2)).squeeze(2)
        residual_batch = y_batch - Dx_batch
        residual_norm_sq = torch.sum(torch.abs(residual_batch)**2, dim=1)
        
        # Trace of inverse directly from precomputed Sigma_y_inv
        # diagonal shape: (batch_size, M) -> sum dim 1 -> (batch_size,)
        trace_inv = torch.sum(torch.real(torch.diagonal(Sigma_y_inv, dim1=1, dim2=2)), dim=1)
        
        numerator = residual_norm_sq + sigma * (M - sigma * trace_inv)
        sigma = torch.clamp(numerator / M, min=eps)


    return x

