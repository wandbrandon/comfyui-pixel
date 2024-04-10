import torch

# # Example tensor
# x = (torch.randn((1, 2, 2, 3)) * 100).int()

# print("OG", x)

# x = torch.reshape(x, (1, 1, -1, 3))
# x = torch.permute(x, (0, 3, 1, 2))

# # Number of times to duplicate
# n = 4

# # Duplicate the tensor n times
# # The new shape will be [n, A, B]
# x_duplicated = x.repeat(1, 1, 1, n)

# print("Original tensor:\n", x, x.shape)
# print("Duplicated tensor:\n", x_duplicated)
# print("Duplicated tensor shape:", x_duplicated.shape)

# # Duplicate the tensor n times
# # The new shape will be [n, A, B]
# x_duplicate_interleaved = x.repeat_interleave(n, dim=3)

# print("Duplicated tensor y:\n", x_duplicate_interleaved)
# print("Duplicated tensor shape y:", x_duplicate_interleaved.shape)

# L1 = x_duplicated[:, 0, :, :]
# A1 = x_duplicated[:, 1, :, :]
# B1 = x_duplicated[:, 2, :, :]
# L2 = x_duplicate_interleaved[:, 0, :, :]
# A2 = x_duplicate_interleaved[:, 1, :, :]
# B2 = x_duplicate_interleaved[:, 2, :, :]

# print(f"L1: {L1} -> L2: {L2}")


def unflatten_upper_triangular(flattened_tensor):
    # Step 1: Determine the size of the full matrix
    n = flattened_tensor.size(0)
    N = int(((8 * n + 1) ** 0.5 + 1) / 2)  # Solving the quadratic equation for N

    # Step 2: Create the full matrix
    full_matrix = torch.zeros(
        (N, N), dtype=flattened_tensor.dtype, device=flattened_tensor.device
    )

    # Indices of the upper triangular part (excluding the diagonal)
    triu_indices = torch.triu_indices(N, N, offset=1)

    # Fill the upper triangular part
    full_matrix[triu_indices[0], triu_indices[1]] = flattened_tensor

    # Since it's a symmetric matrix, copy the upper triangular part to the lower triangular part
    full_matrix = full_matrix + full_matrix.t()

    return full_matrix


# Example
# For a 4x4 matrix, there are 6 elements above the diagonal
flattened_tensor = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)

# Reconstruct the full matrix
full_matrix = unflatten_upper_triangular(flattened_tensor)

print(full_matrix)
