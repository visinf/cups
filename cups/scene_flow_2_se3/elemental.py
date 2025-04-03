import torch


def multiply_matrix_vector(matrix, vector, cdim=1):
    # vector torch.Tensor:    S1 x ... Sb x Sc x ... (with c=cdim)
    # matrix torch.Tensor: S1 x ... Sb x  Sc x Sc

    vector_shape = torch.LongTensor(list(vector.shape))
    C = vector_shape[cdim]
    SB = vector_shape[:cdim]
    matrix_SB = torch.LongTensor(list(matrix.shape))[:cdim]
    SA = vector_shape[cdim + 1 :]

    # insert 1 x 3 at second last, delete cdim
    vector_op_shape = torch.cat((SB, torch.LongTensor([1, C]), SA))
    vector_op_size = torch.Size(vector_op_shape)
    # S1 x ... x 1 x 3

    # insert 3 x 3 at second last, delete cdim
    matrix_op_shape = torch.cat((matrix_SB, torch.LongTensor([C, C]), torch.ones(len(SA), dtype=torch.long)))
    matrix_op_size = torch.Size(matrix_op_shape)

    # broadcast over second last axis, and sum over last axis
    product = (matrix.reshape(matrix_op_size) * vector.reshape(vector_op_size)).sum(dim=cdim + 1)

    return product


def multiply_matrix_matrix(matrix1, matrix2):
    # matrix1 torch.Tensor: S1 x ... Sb1 x Sm1 x Sc (with c1=c1dim, c2=c2dim)
    # matrix2 torch.Tensor: S1 x ... Sb2 x Sc x Sm2

    matrix1_shape = torch.LongTensor(list(matrix1.shape))
    matrix2_shape = torch.LongTensor(list(matrix2.shape))

    matrix1_op_shape = torch.cat((matrix1_shape, torch.LongTensor([1])))
    matrix1_op_size = torch.Size(matrix1_op_shape)
    matrix2_op_shape = torch.cat((matrix2_shape[:-2], torch.LongTensor([1]), matrix2_shape[-2:]))
    matrix2_op_size = torch.Size(matrix2_op_shape)

    # broadcast over second last axis, and sum over last axis
    product = (matrix1.reshape(matrix1_op_size) * matrix2.reshape(matrix2_op_size)).sum(dim=-2)

    return product
