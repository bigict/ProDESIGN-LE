import torch
from torch import einsum
from einops import rearrange


def vec2rotation(vec):
    """Convert a set of 3D vectors into a rotation matrix.
    
    This function takes three 3D points (forming a triangle) and computes an orthonormal
    basis (rotation matrix) where:
    - e1 points along the vector from point 1 to point 2
    - e2 lies in the plane of the triangle
    - e3 is perpendicular to the plane (cross product of e1 and e2)
    
    Args:
        vec: Tensor of shape (..., 3, 3) where each (..., i, :) represents a 3D point
        
    Returns:
        Rotation matrix tensor of shape (..., 3, 3) representing the orthonormal basis
    """
    # Compute vectors between points
    v1 = vec[..., 2, :] - vec[..., 1, :]  # Vector from point 1 to point 2
    v2 = vec[..., 0, :] - vec[..., 1, :]  # Vector from point 1 to point 0
    
    # Create first basis vector (e1) by normalizing v1
    e1 = v1 / vector_robust_norm(v1, dim=-1, keepdim=True)
    
    # Create second basis vector (e2) by Gram-Schmidt orthogonalization
    # Project v2 onto e1 and subtract to get orthogonal component
    u2 = v2 - e1 * rearrange(einsum('...L,...L->...', e1, v2), '...->...()')
    e2 = u2 / vector_robust_norm(u2, dim=-1, keepdim=True)
    
    # Create third basis vector (e3) via cross product
    e3 = torch.cross(e1, e2, dim=-1)
    
    # Stack basis vectors to form rotation matrix
    return torch.stack((e1, e2, e3), dim=-1)

def vec2transform(vec):
    """Convert a set of 3D vectors into a 4x4 transformation matrix.
    
    The transformation matrix combines rotation and translation:
    - Rotation comes from vec2rotation
    - Translation comes from the second point in the input vectors
    
    Args:
        vec: Tensor of shape (..., 3, 3) where each (..., i, :) represents a 3D point
        
    Returns:
        4x4 transformation matrix tensor of shape (..., 4, 4)
    """
    result = torch.zeros(*vec.shape[:-2], 4, 4, device=vec.device)
    result[..., :3, :3] = vec2rotation(vec)  # Rotation component
    result[..., :3, 3] = vec[..., 1, :]     # Translation component
    result[..., 3, 3] = 1                   # Homogeneous coordinate
    return result

def transform2feat(vec):
    """Flatten a transformation matrix into a feature vector.
    
    Args:
        vec: Transformation matrix tensor of shape (..., 4, 4)
        
    Returns:
        Flattened feature tensor of shape (..., 9) containing only rotation components
    """
    return rearrange(vec[..., :3, :], '... X Y -> ... (X Y)')

def transform_invert(transform):
    """Compute the inverse of a transformation matrix.
    
    For a transformation matrix T = [R | t], the inverse is:
    T⁻¹ = [Rᵀ | -Rᵀt]
    
    Args:
        transform: Transformation matrix tensor of shape (..., 4, 4)
        
    Returns:
        Inverse transformation matrix tensor of shape (..., 4, 4)
    """
    result = torch.zeros(*transform.shape[:-2], 4, 4, device=transform.device)
    result[..., :3, :3] = transform[..., :3, :3].transpose(-1, -2)  # Transpose rotation
    result[..., :3, -1] = einsum('...ij,...j->...i',
        -result[..., :3, :3], transform[..., :3, -1])  # Compute inverse translation
    result[..., 3, 3] = 1  # Homogeneous coordinate
    return result

def vec2homo(vec):
    """Convert 3D vectors to homogeneous coordinates.
    
    Args:
        vec: Tensor of shape (..., 3) representing 3D vectors
        
    Returns:
        Tensor of shape (..., 4) with homogeneous coordinates (x,y,z,1)
    """
    result = torch.ones(*vec.shape[:-1], 4, device=vec.device)
    result[..., :3] = vec  # Copy x,y,z coordinates
    return result

def vector_robust_norm(vec, epison=1e-8, **kargs):
    """Compute vector norm with added epsilon for numerical stability.
    
    Args:
        vec: Input tensor
        epison: Small value to add to norm for stability
        **kargs: Additional arguments to torch.linalg.vector_norm
        
    Returns:
        Tensor of vector norms with added epsilon
    """
    return torch.linalg.vector_norm(vec, **kargs) + epison
