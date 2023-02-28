"""
Official implementation was rewrite into torch.
[https://github.com/mercedes-benz/selfsupervised_flow/blob/a51cb5dddb1ab409ce18d3d22d650d712fa60ae1/unsup_flow/datasets/kitti_sf/weighted_pc_alignment.py#L9]
"""


import torch
import torch.nn as nn

def weighted_pc_alignment(cloud_t0, cloud_t1, weights):
    """
    Computes the weighted point cloud alignment between two point clouds.

    Args:
    - cloud_t0 (torch.Tensor): The first point cloud tensor of shape (m, n).
    - cloud_t1 (torch.Tensor): The second point cloud tensor of shape (m, n).
    - weights (torch.Tensor): The weights tensor of shape (n,).

    Returns:
    - R (torch.Tensor): The rotation matrix of shape (3, 3).
    - c (float): The scaling factor.
    - t (torch.Tensor): The translation vector of shape (3,).
    """
    m, n = cloud_t0.shape
    #weights = weights
    cum_wts = torch.sum(weights)

    X_wtd = cloud_t0 * weights
    Y_wtd = cloud_t1 * weights

    mx_wtd = torch.sum(X_wtd, dim=1) / cum_wts
    my_wtd = torch.sum(Y_wtd, dim=1) / cum_wts
    print("mx_wtd =\n", mx_wtd)

    Xc = cloud_t0 - torch.tile(mx_wtd, (n, 1)).T
    Yc = cloud_t1 - torch.tile(my_wtd, (n, 1)).T
    print("Xc=\n", Xc)


    Sxy_wtd = torch.mm(Yc * weights, Xc.T) / cum_wts
    print("Sxy_wtd matrix=\n", Sxy_wtd)


    U, D, V = torch.linalg.svd(Sxy_wtd)
    V = V.T.clone()
    r = torch.linalg.matrix_rank(Sxy_wtd)

    print("U matrix=\n", U)
    print("D matrix=\n", D)
    print("V matrix=\n", V)

    S = torch.eye(m)

    if r > (m - 1):
        if torch.linalg.det(Sxy_wtd) < 0.0:
            S[m - 1, m - 1] = -1.0
    elif r == m - 1:
        det_mul = torch.linalg.det(U) * torch.linalg.det(V)
        if torch.isclose(det_mul, torch.tensor(-1.0)):
            S[m - 1, m - 1] = -1.0
    else:
        raise RuntimeError("Rank deterioration!")

    R = torch.mm(torch.mm(U, S), V.T)

    c = 1.0
    t = my_wtd - c * torch.mm(R, mx_wtd[:, None]).flatten()

    T = torch.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return R, c, t


class WeightedKabschAlgorithm(nn.Module):
    """
    PyTorch implementation of weighted point cloud alignment.

    Args:
        use_epsilon_on_weights (bool, optional): Whether to add a small epsilon to weights. Default is False.

    Inputs:
        cloud_t0 (torch.Tensor): Input point cloud at time t0, shape (batch_size, num_points, 3).
        cloud_t1 (torch.Tensor): Input point cloud at time t1, shape (batch_size, num_points, 3).
        weights (torch.Tensor): Weights for each point, shape (batch_size, num_points).

    Returns:
        T (torch.Tensor): 4x4 transformation matrix, shape (batch_size, 4, 4).
        not_enough_points (torch.Tensor): Boolean tensor indicating if there were not enough points to compute the
            transformation matrix, shape (batch_size,).
    """

    def __init__(self, use_epsilon_on_weights=False):
        super().__init__()
        self.use_epsilon_on_weights = use_epsilon_on_weights

    def forward(self, cloud_t0, cloud_t1, weights):
        dims = 3
        assert cloud_t0.shape[-1] == dims
        assert cloud_t1.shape[-1] == dims
        assert len(weights.shape) == 2

        assert (weights >= 0.0).all(), "Negative weights found"
        if self.use_epsilon_on_weights:
            weights += torch.finfo(weights.dtype).eps
            count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
            not_enough_points = count_nonzero_weighted_points < 3
        else:
            # Add eps if not enough points with weight over zero
            count_nonzero_weighted_points = torch.sum(weights > 0.0, dim=-1)
            not_enough_points = count_nonzero_weighted_points < 3
            eps = not_enough_points.float() * torch.finfo(weights.dtype).eps
            weights += eps.unsqueeze(-1)

        cum_wts = torch.sum(weights, dim=-1)

        #X_wtd = torch.sum(cloud_t0.unsqueeze(-2) * weights.unsqueeze(-1), dim=-3)
        #Y_wtd = torch.sum(cloud_t1.unsqueeze(-2) * weights.unsqueeze(-1), dim=-3)

        X_wtd = cloud_t0 * weights.unsqueeze(-1)
        Y_wtd = cloud_t1 * weights.unsqueeze(-1)

        mx_wtd = torch.sum(X_wtd, dim=1) / cum_wts.unsqueeze(0)
        my_wtd = torch.sum(Y_wtd, dim=1) / cum_wts.unsqueeze(0)

        Xc = cloud_t0 - mx_wtd.unsqueeze(0)
        Yc = cloud_t1 - my_wtd.unsqueeze(0)

        # Covariance matrix
        Sxy_wtd = ((Yc * weights.unsqueeze(-1)).transpose(1,2) @ Xc) / cum_wts

        U, D, V = torch.svd(Sxy_wtd)

        # TODO #
        m = 3
        r = torch.linalg.matrix_rank(Sxy_wtd)
        S = torch.eye(m)
        if r > (m - 1):
            if torch.linalg.det(Sxy_wtd) < 0.0:
                S[m - 1, m - 1] = -1.0
        elif r == m - 1:
            det_mul = torch.linalg.det(U) * torch.linalg.det(V)
            if torch.isclose(det_mul, torch.tensor(-1.0)):
                S[m - 1, m - 1] = -1.0
        else:
            raise RuntimeError("Rank deterioration!")
        # TODO #

        R = torch.einsum("boc,bic->boi", torch.einsum("boc,bic->boi", U, S.unsqueeze(0)), V)
        #R = torch.einsum("boc,bic->boi", U, V)

        # c = torch.trace(torch.matmul(torch.diag(D), S)) / sx
        # c = 1.0
        t = my_wtd - torch.einsum("boc,bc->bo", R, mx_wtd)
        # t = my_wtd - c * torch.matmul(R, mx_wtd)

        # T = torch.eye(dims + 1, batch_shape=torch.shape(weights)[:1])
        R = torch.cat([R, torch.zeros_like(R[:, :1, :])], dim=1)
        t = torch.cat([t, torch.ones_like(t[:, :1])], dim=-1)
        T = torch.cat([R, t[:, :, None]], dim=-1)

        #assert T.dtype == torch.float64

        return T, not_enough_points  # R, t



if __name__ == "__main__":
    # Run an example test
    # We have 3 points in 3D. Every point is a column vector of this matrix A
    A = torch.tensor(
        [
            [0.57215, 0.37512, 0.37551, 1.57215, 1.37512, 1.37551],
            [0.23318, 0.86846, 0.98642, 1.23318, 1.86846, 1.98642],
            [0.79969, 0.96778, 0.27493, 1.79969, 1.96778, 1.27493],
        ]
    )
    # Deep copy A to get B
    B = A.clone()
    # and sum a translation on z axis (3rd row) of 10 units
    B[2, :3] = B[2, :3] + 10
    B[2, 3:] = B[2, 3:] + 11
    weights = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])


    # Reconstruct the transformation with weighted_pc_alignment
    R, c, t = weighted_pc_alignment(A, B, weights)
    print("Rotation matrix=\n", R, "\nScaling coefficient=", c, "\nTranslation vector=", t)

    kabsch = WeightedKabschAlgorithm()
    T, not_points = kabsch(A.unsqueeze(0).transpose(1,2), B.unsqueeze(0).transpose(1,2), weights.unsqueeze(0))
    print("Transformation matrix=\n", T)
