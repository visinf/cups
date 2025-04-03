import torch
from scipy.optimize import linear_sum_assignment


def hungarian_one_to_one(cost_matrix):
    # M x N with M >= N
    # M workers, N jobs
    np_cost_matrix = cost_matrix.detach().cpu().numpy()

    np_row_ind, np_col_ind = linear_sum_assignment(np_cost_matrix)

    col_ind = torch.from_numpy(np_col_ind).to(cost_matrix.device)
    row_ind = torch.from_numpy(np_row_ind).to(cost_matrix.device)

    return row_ind, col_ind
