import pytorch3d.transforms as t3d
import torch

#############################
### NEW SE3 LAYER (se3toSE3 with some additional transformation for the translation vector)


# Create a skew-symmetric matrix "S" of size [(Bk) x 3 x 3] (passed in) given a [(Bk) x 3] vector
def create_skew_symmetric_matrix(vector):
    # Create the skew symmetric matrix:
    # [0 -z y; z 0 -x; -y x 0]
    N = vector.size(0)
    output = vector.view(N, 3, 1).expand(N, 3, 3).clone()
    output[:, 0, 0] = 0
    output[:, 1, 1] = 0
    output[:, 2, 2] = 0
    output[:, 0, 1] = -vector[:, 2]
    output[:, 1, 0] = vector[:, 2]
    output[:, 0, 2] = vector[:, 1]
    output[:, 2, 0] = -vector[:, 1]
    output[:, 1, 2] = -vector[:, 0]
    output[:, 2, 1] = vector[:, 0]
    return output


# Compute the rotation matrix R & translation vector from the axis-angle parameters using Rodriguez's formula:
# (R = I + (sin(theta)/theta) * K + ((1-cos(theta))/theta^2) * K^2)
# (t = I + (1-cos(theta))/theta^2 * K + ((theta-sin(theta)/theta^3) * K^2
# where K is the skew symmetric matrix based on the un-normalized axis & theta is the norm of the input parameters
# From Wikipedia: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
# Eqns: 77-84 (PAGE: 10) of http://ethaneade.com/lie.pdf
def se3ToRt(input):
    eps = 1e-12
    # Get the un-normalized axis and angle
    N = input.size(0)
    axis = input.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle = torch.sqrt(angle2)  # Angle

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute sines
    S = torch.sin(angle) / angle
    S.masked_fill_(angle2.lt(eps), 1)  # sin(0)/0 ~= 1

    # Compute cosines
    C = (1 - torch.cos(angle)) / angle2
    C.masked_fill_(angle2.lt(eps), 0)  # (1 - cos(0))/0^2 ~= 0

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    rot = (
        torch.eye(3).view(1, 3, 3).repeat(N, 1, 1).type_as(input)
    )  # R = I (avoid use expand as it does not allocate new memory)
    rot += K * S.expand(N, 3, 3)  # R = I + (sin(theta)/theta)*K
    rot += K2 * C.expand(N, 3, 3)
    return rot
    """# Check dimensions

    bsz, nse3, ndim = input.size()
    N = bsz * nse3
    eps = 1e-12
    assert ndim == 6

    # Trans | Rot params
    input_v = input.view(N, ndim, 1)
    rotparam = input_v.narrow(1, 3, 3)  # so(3)
    transparam = input_v.narrow(1, 0, 3)  # R^3

    # Get the un-normalized axis and angle
    axis = rotparam.clone().view(N, 3, 1)  # Un-normalized axis
    angle2 = (axis * axis).sum(1).view(N, 1, 1)  # Norm of vector (squared angle)
    angle = torch.sqrt(angle2)  # Angle
    small = (
        angle2 < eps
    ).detach()  # Don't need gradient w.r.t this operation (also because of pytorch error: https://discuss.pytorch.org/t/get-error-message-maskedfill-cant-differentiate-the-mask/9129/4)

    # Create Identity matrix
    I = angle2.expand(N, 3, 3).clone()
    I[:] = 0  # Fill will all zeros
    I[:, 0, 0], I[:, 1, 1], I[:, 2, 2] = 1.0, 1.0, 1.0  # Diagonal elements = 1

    # Compute skew-symmetric matrix "K" from the axis of rotation
    K = create_skew_symmetric_matrix(axis)
    K2 = torch.bmm(K, K)  # K * K

    # Compute A = (sin(theta)/theta)
    A = torch.sin(angle) / angle
    A[small] = 1.0  # sin(0)/0 ~= 1

    # Compute B = (1 - cos(theta)/theta^2)
    B = (1 - torch.cos(angle)) / angle2
    B[small] = 1 / 2  # lim 0-> 0 (1 - cos(0))/0^2 = 1/2

    # Compute C = (theta - sin(theta))/theta^3
    C = (1 - A) / angle2
    C[small] = 1 / 6  # lim 0-> 0 (0 - sin(0))/0^3 = 1/6

    # Compute the rotation matrix: R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2) * K^2
    R = I + K * A.expand(N, 3, 3) + K2 * B.expand(N, 3, 3)

    # Compute the translation tfm matrix: V = I + ((1-cos(theta))/theta^2) * K + ((theta-sin(theta))/theta^3)*K^2
    V = I + K * B.expand(N, 3, 3) + K2 * C.expand(N, 3, 3)

    # Compute translation vector
    t = torch.bmm(V, transparam)  # V*u

    # Final tfm
    return torch.cat([R, t], 2).view(bsz, nse3, 3, 4).clone()  # B x K x 3 x 4
    """


def so3d_to_so3_mat(so3d):
    # so3_3d Bx3
    B, _ = so3d.shape

    angle1d = torch.norm(so3d, dim=1, p=2)
    axis3d = so3d / (angle1d.unsqueeze(-1) + 1e-10)
    cos_angle1d = torch.cos(angle1d)
    sin_angle1d = torch.sin(angle1d)

    so3_mat = (
        torch.stack(
            [
                cos_angle1d + axis3d[:, 0] ** 2 * (1 - cos_angle1d),
                axis3d[:, 0] * axis3d[:, 1] * (1 - cos_angle1d) - axis3d[:, 2] * sin_angle1d,
                axis3d[:, 0] * axis3d[:, 2] * (1 - cos_angle1d) + axis3d[:, 1] * sin_angle1d,
                axis3d[:, 1] * axis3d[:, 0] * (1 - cos_angle1d) + axis3d[:, 2] * sin_angle1d,
                cos_angle1d + axis3d[:, 1] ** 2 * (1 - cos_angle1d),
                axis3d[:, 1] * axis3d[:, 2] * (1 - cos_angle1d) - axis3d[:, 0] * sin_angle1d,
                axis3d[:, 2] * axis3d[:, 0] * (1 - cos_angle1d) - axis3d[:, 1] * sin_angle1d,
                axis3d[:, 2] * axis3d[:, 1] * (1 - cos_angle1d) + axis3d[:, 0] * sin_angle1d,
                cos_angle1d + axis3d[:, 2] ** 2 * (1 - cos_angle1d),
            ]
        )
        .reshape(3, 3, -1)
        .permute(2, 0, 1)
    )

    return so3_mat


def se3_6d_to_se3mat(se3_6d, type="exponential"):
    # input shape Bx6 or BxKx6
    K = None
    if len(se3_6d.shape) == 3:
        K = se3_6d.shape[1]
        se3_6d = se3_6d.view(-1, 6)
    B, _ = se3_6d.shape
    dtype = se3_6d.dtype
    device = se3_6d.device

    se3mat = torch.zeros(size=(B, 4, 4), dtype=dtype, device=device)

    if type == "axisangle":
        se3mat[:, :3, :3] = so3d_to_so3_mat(se3_6d[:, :3])
        se3mat[:, :3, 3] = se3_6d[:, 3:]
    elif type == "exponential":
        se3mat[:, :3, :3] = t3d.so3_exponential_map(se3_6d[:, :3])
        se3mat[:, :3, 3] = se3_6d[:, 3:]
    elif type == "hind4sight":
        # se3mat[:, :3, :4] = se3ToRt(se3_6d[:, :].unsqueeze(1))
        se3mat[:, :3, :3] = se3ToRt(se3_6d[:, :3])
        se3mat[:, :3, 3] = se3_6d[:, 3:]

    # transl_3d = torch.matmul(se3mat[:, :3, :3], se3_6d[:, 3:].unsqueeze(-1)).squeeze(-1)
    # se3mat[:, :3, 3] = transl_3d

    # se3mat[:, :3, :3] = torch.eye(3)

    se3mat[:, 3, 3] = 1.0

    if K is not None:
        se3mat = se3mat.view(-1, K, 4, 4)

    return se3mat


def pts3d_rotate(pts3d, rot, mask=None):
    # pts3d: BxNx3 or Bx3xHxW
    # se3mat: Bx4x4 or BxKx4x4
    # masks: BxKxHxW
    B = len(rot)
    se3mat = torch.eye(4, dtype=rot.dtype, device=rot.device)[None, :, :].repeat(B, 1, 1)

    se3mat[:, :3, :3] = rot

    return pts3d_transform(pts3d, se3mat, mask=mask)


def pts3d_transform(pts3d, se3mat, mask=None):
    # pts3d: BxNx3 or Bx3xHxW
    # se3mat: Bx4x4 or BxKx4x4
    # masks: BxKxHxW
    pts3d_shape = pts3d.shape
    dtype = pts3d.dtype
    device = pts3d.device

    if len(pts3d_shape) == 4:
        B, _, H, W = pts3d_shape
        pts3d = pts3d.reshape(B, 3, H * W).permute(0, 2, 1)

    B, N, _ = pts3d.shape
    # se3mat Bx4x4
    # pts3d BxNx3
    pts4d = torch.cat((pts3d, torch.ones(size=(B, N, 1), dtype=dtype, device=device)), dim=2)
    if mask is None:
        # se3mat: Bx1x4x4, pts3d: BxNx4x1
        pts4d_ftf = torch.matmul(se3mat.unsqueeze(1), pts4d.unsqueeze(-1)).squeeze(-1)
        # pts4d: BxNx4
        pts3d_ftf = pts4d_ftf[:, :, :3]
    else:
        # se3mat: BxKx1x4x4, pts3d: Bx1xNx4x1
        # se3mat[:, 0] = torch.eye(4, dtype=se3mat.dtype, device=se3mat.device)
        # se3mat = se3mat * 0.
        pts4d_ftf = torch.matmul(se3mat.unsqueeze(2), pts4d.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        # pts4d: BxKxNx4 -> BxKxNx3
        pts3d_ftf = pts4d_ftf[:, :, :, :3]
        B, K, H, W = mask.shape
        # B x K x H x W
        mask = mask.reshape(B, K, H * W).unsqueeze(-1)
        # BxKxNx1 * BxKxNx3
        pts3d_ftf = pts3d + torch.sum(mask * (pts3d_ftf - pts3d.unsqueeze(1)), dim=1)

    if len(pts3d_shape) == 4:
        pts3d_ftf = pts3d_ftf.permute(0, 2, 1).reshape(B, 3, H, W)

    return pts3d_ftf


def pts3d_transform_obj_ego(pts3d, se3mat, mask, egomotion_addition=True):

    # transform with egomotion

    if egomotion_addition:
        pts3d = pts3d_transform(pts3d, se3mat[:, 0])

    if mask is not None:
        # transform with masks + obj_motions
        if egomotion_addition:
            se3mat = se3mat.clone()
            se3mat[:, 0] = torch.eye(4, dtype=se3mat.dtype, device=se3mat.device)
        pts3d = pts3d_transform(pts3d, se3mat[:, 0:], mask[:, 0:])

    return pts3d


def sflow_via_transform(pts3d, se3mat, mask, egomotion_addition):
    pts3d_ftf = pts3d_transform_obj_ego(pts3d, se3mat, mask, egomotion_addition=egomotion_addition)
    flow = pts3d_ftf - pts3d.detach()
    return flow, mask
