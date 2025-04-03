import torch


def cluster(pair_rigid, masks_rigid_range, masks_cores, num_steps, pair_neighbor=None, masks_neighbor_range=None):
    # pair_rigid, pair_neighbor: N x N
    for i in range(num_steps - 1):
        masks_rigid_range, masks_neighbor_range, masks_rigid_cores = random_core_accumulation_step(
            pair_rigid,
            masks_rigid_range,
            masks_cores,
            pair_neighbor=pair_neighbor,
            masks_neighbor_range=masks_neighbor_range,
        )

    return masks_rigid_range, masks_cores


def random_core_accumulation_step(
    pair_rigid, masks_rigid_range, masks_cores, pair_neighbor=None, masks_neighbor_range=None
):

    device = masks_rigid_range.device
    K, N = masks_rigid_range.shape
    masks_pot_cores = masks_rigid_range * (~masks_cores)
    if masks_neighbor_range is not None and pair_neighbor is not None:
        masks_pot_cores *= masks_neighbor_range
    num_pot_cores = masks_pot_cores.sum(dim=1)
    rand_core_ids_offset = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(num_pot_cores, dim=0),
        ),
        dim=0,
    )[:-1]

    rand_core_ids = rand_core_ids_offset + (torch.rand(K).to(device) * num_pot_cores).floor().long()
    rand_core_ids = rand_core_ids[num_pot_cores > 0]

    rand_cores_grid_ids = masks_pot_cores.nonzero()[rand_core_ids]
    mask_new_cores = torch.zeros_like(masks_cores)
    mask_new_cores[rand_cores_grid_ids[:, 0], rand_cores_grid_ids[:, 1]] = True

    masks_cores += mask_new_cores
    masks_rigid_range[num_pot_cores > 0] *= pair_rigid[rand_cores_grid_ids[:, 1]]
    if masks_neighbor_range is not None and pair_neighbor is not None:
        masks_neighbor_range[num_pot_cores > 0] += pair_neighbor[rand_cores_grid_ids[:, 1]]
    return masks_rigid_range, masks_neighbor_range, masks_cores
