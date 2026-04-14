import torch
import torch.nn.functional as F

def centered_cosine_similarity(x, y):
    """
    Computes centered cosine similarity between two feature representations.
    Usually applied to 1D tensors (e.g., feature vectors of an image).
    """
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    return F.cosine_similarity(x_c, y_c, dim=-1)

def compute_gram_matrix(x):
    """
    Computes the Gram matrix X @ X^T
    """
    x = x.view(x.size(0), -1)
    return torch.matmul(x, x.t())

def center_gram_matrix(gram_matrix):
    """
    Centers the Gram matrix.
    H = I - 1/n * 1 1^T
    K_centered = H K H
    """
    n = gram_matrix.size(0)
    H = torch.eye(n, device=gram_matrix.device) - torch.ones((n, n), device=gram_matrix.device) / n
    return torch.matmul(torch.matmul(H, gram_matrix), H)

def linear_cka(x, y):
    """
    Computes Linear Centered Kernel Alignment (CKA) between two sets of features.
    x and y should be of shape (N, D1) and (N, D2) where N is the number of samples.
    """
    gram_x = compute_gram_matrix(x)
    gram_y = compute_gram_matrix(y)
    
    gram_x_c = center_gram_matrix(gram_x)
    gram_y_c = center_gram_matrix(gram_y)
    
    # Trace of product of centered Gram matrices is the Frobenius inner product
    scaled_hsic = torch.sum(gram_x_c * gram_y_c)
    norm_x = torch.sqrt(torch.sum(gram_x_c * gram_x_c))
    norm_y = torch.sqrt(torch.sum(gram_y_c * gram_y_c))
    
    return scaled_hsic / (norm_x * norm_y)

def rsa(x, y):
    """
    Computes Representational Similarity Analysis (RSA) between two sets of features.
    It computes the Representation Data Matrices (RDMs) using Pearson correlation 
    (1 - corr) and then computes the Pearson correlation between the RDMs.
    """
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    
    # Compute centered representations for correlation
    x_c = x - x.mean(dim=1, keepdim=True)
    y_c = y - y.mean(dim=1, keepdim=True)
    
    # Compute correlation matrices (N x N)
    x_norms = torch.norm(x_c, dim=1, keepdim=True)
    y_norms = torch.norm(y_c, dim=1, keepdim=True)
    
    # Add epsilon to prevent division by zero
    eps = 1e-8
    rdm_x = 1 - torch.matmul(x_c, x_c.t()) / (torch.matmul(x_norms, x_norms.t()) + eps)
    rdm_y = 1 - torch.matmul(y_c, y_c.t()) / (torch.matmul(y_norms, y_norms.t()) + eps)
    
    # Get upper triangular elements (excluding diagonal)
    n = rdm_x.size(0)
    idx = torch.triu_indices(n, n, offset=1)
    
    rdm_x_vec = rdm_x[idx[0], idx[1]]
    rdm_y_vec = rdm_y[idx[0], idx[1]]
    
    # Compute Pearson correlation between the RDMs
    rdm_x_vec_c = rdm_x_vec - rdm_x_vec.mean()
    rdm_y_vec_c = rdm_y_vec - rdm_y_vec.mean()
    
    num = torch.sum(rdm_x_vec_c * rdm_y_vec_c)
    den = torch.norm(rdm_x_vec_c) * torch.norm(rdm_y_vec_c) + eps
    
    return num / den

