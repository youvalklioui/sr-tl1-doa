import torch


def prox_l1_norm(x, beta):
    "Returns the proximal operator for the L1 norm evaluated at x with parameter beta"
        
    zeros = torch.zeros(x.size(), device=x.device)
    prox = torch.exp(1j * x.angle()) * torch.max(
        torch.abs(x) - beta, zeros
    )
    return prox


def prox_l2_norm(x,beta):
    "Minimization of L1-2 for Compressed Sensing (https://epubs.siam.org/doi/epdf/10.1137/140952363)"
    "Returns the proximal operator for the L2 norm evaluated at x with parameter beta"
        
    zeros = torch.zeros(x.size(), device=x.device)
    prox = torch.exp(1j * x.angle()) * torch.max(
        torch.abs(x) - beta, zeros
    )
    return prox



def subgradient_l2_norm(x):
    "Returns the subgradient of the L2 norm evaluated at x"

    norm = torch.norm(x, dim=0, keepdim=True)
    
    subgrad = x / (2 * norm)
    subgrad = torch.where(norm == 0, torch.zeros_like(x), subgrad)

    return subgrad


def subgradient_mcp_norm(x,gamma):
    "Nearly unbiased variable selection under minimax concave penalty': https://arxiv.org/abs/1002.4734)"
    "The MCP norm can be expressed as difference of convex functions under the form: "
    "MCP(x, gamma) = ||x||_{1} - ( ||x||_{1} + 1 / (2 * gamma) * sum ( max( |xi| - gamma, 0) )^{2} )"
    "The Wirtinger gradient of the second term above is what this function returns."
    
    subgrad = torch.exp(1j * x.angle()) * torch.where(torch.abs(x) <= gamma, x / (2 * gamma), torch.ones_like(x)/2)

    return subgrad



def subgradient_tl1_norm(x, alpha = 1):
        
        "Minimization of Transformed L1 Penalty: Theory, Difference of Convex Function Algorithm, and Robust Application in Compressed Sensing (https://arxiv.org/abs/1411.5735)"

        "Returns the subgradient of the second term in the DC decomposition of the TL1 norm with parameter alpha."

        subgrad =  (alpha + 1) / alpha * (alpha + torch.abs(x) / 2) / (alpha + torch.abs(x))**2  * x   
           
        return subgrad


def subgradient(framework, x, hyperparameter = 1):

    "Wrapper function to call the appropriate subgradient function based on the framework."

    if framework in ['TL1', 'SR-TL1']:
        return subgradient_tl1_norm(x, alpha = hyperparameter)
    elif framework in ['L1L2']:
        return subgradient_l2_norm(x)
    elif framework in ['MCP']:
        return subgradient_mcp_norm(x, gamma = hyperparameter)

    else:
        raise ValueError(f"Unsupported framework: {framework}")

    
def prox_l1_update(framework, x, kappa = 1, rho = 1, rho2 = 1, hyperparameter = 1):
    "Wrapper function to provide the correct scaling for the proximal operator of L1 norm depending on the framework."

    scale = rho if framework in ['LASSO', 'L1L2', 'TL1', 'MCP'] else rho2

    if framework in ['LASSO', 'L1L2', 'MCP', 'SR-LASSO']:
        return prox_l1_norm(x, kappa/scale)
    elif framework in ['TL1', 'SR-TL1']:
        return prox_l1_norm(x, kappa * (hyperparameter + 1)/(hyperparameter * scale))

