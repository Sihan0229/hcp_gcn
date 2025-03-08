import torch
import numpy as np
from utils.mesh import face_area


def distortion(metric_sphere, metric_surf):
    """
    Compute metric distortion. (area or edge)

    Inputs:
    - metric_sphere: the metric of the (prediced) sphere, (1,|V|) torch.Tensor
    - metric_surf: the metric of the reference WM surface, (1,|V|) torch.Tensor
    - 如果球面和 WM 表面度量在尺度上不同，我们希望找到一个最优的𝛽,使得整体误差最小。
    Returns:
    - distort: the metric distortion (RMSD), torch.float
    """

    # beta = (metric_sphere * metric_surf).mean() /  (metric_surf**2).mean()
    # distort = ((metric_sphere - beta*metric_surf)**2).mean()
    beta = (metric_sphere * metric_surf).mean() /  (metric_sphere**2).mean()
    distort = ((beta*metric_sphere - metric_surf)**2).mean().sqrt()
    return distort


def edge_distortion(vert_sphere, vert_surf, edge):
    """
    Compute edge distortion.

    Inputs:
    - vert_sphere: the vertices of the (prediced) sphere, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - edge: the edge list of the mesh, (2,|E|) torch.LongTensor
    
    Returns:
    - edge distortion, torch.float
    """
    # compute edge length
    edge_len_sphere = (vert_sphere[:,edge[0]] -\
                       vert_sphere[:,edge[1]]).norm(dim=-1)
    edge_len_surf = (vert_surf[:,edge[0]] -\
                     vert_surf[:,edge[1]]).norm(dim=-1)
    return distortion(edge_len_sphere, edge_len_surf)


def area_distortion(vert_sphere, vert_surf, face):
    """
    Compute area distortion.

    Inputs:
    - vert_sphere: the vertices of the (prediced) sphere, (1,|V|) torch.Tensor
    - vert_surf: the vertices of the reference WM surface, (1,|V|) torch.Tensor
    - face: the mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area distortion, torch.float
    """
    
    # compute face area
    area_sphere = face_area(vert_sphere, face)
    area_surf = face_area(vert_surf, face)
    return distortion(area_sphere, area_surf)

def chamfer_distance(pc1, pc2):
    """
    Compute Chamfer Distance between two point clouds.

    Inputs:
    - pc1: (N, 3) torch.Tensor, predicted points
    - pc2: (M, 3) torch.Tensor, reference points
    
    Returns:
    - chamfer_dist: Chamfer distance, torch.float
    """
    dist_matrix = torch.cdist(pc1, pc2)  # 计算点对之间的欧式距离矩阵
    min_dist1 = dist_matrix.min(dim=1)[0]  # pc1 到 pc2 的最近点距离
    min_dist2 = dist_matrix.min(dim=0)[0]  # pc2 到 pc1 的最近点距离
    return min_dist1.mean() + min_dist2.mean()


def hausdorff_distance(pc1, pc2):
    """
    Compute Hausdorff Distance between two point clouds.

    Inputs:
    - pc1: (N, 3) torch.Tensor, predicted points
    - pc2: (M, 3) torch.Tensor, reference points
    
    Returns:
    - hausdorff_dist: Hausdorff distance, torch.float
    """
    dist_matrix = torch.cdist(pc1, pc2)
    min_dist1 = dist_matrix.min(dim=1)[0]  # pc1 到 pc2 的最近点距离
    min_dist2 = dist_matrix.min(dim=0)[0]  # pc2 到 pc1 的最近点距离
    return torch.max(torch.max(min_dist1), torch.max(min_dist2))
