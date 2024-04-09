import torch

# Example tensor
x = (torch.randn((1, 2, 2, 3)) * 100).int()

print("OG", x)

x = torch.reshape(x, (1, 1, -1, 3))
x = torch.permute(x, (0, 3, 1, 2))

# Number of times to duplicate
n = 4

# Duplicate the tensor n times
# The new shape will be [n, A, B]
x_duplicated = x.repeat(1, 1, 1, n)

print("Original tensor:\n", x, x.shape)
print("Duplicated tensor:\n", x_duplicated)
print("Duplicated tensor shape:", x_duplicated.shape)

# Duplicate the tensor n times
# The new shape will be [n, A, B]
x_duplicate_interleaved = x.repeat_interleave(n, dim=3)

print("Duplicated tensor y:\n", x_duplicate_interleaved)
print("Duplicated tensor shape y:", x_duplicate_interleaved.shape)

L1 = x_duplicated[:, 0, :, :]
A1 = x_duplicated[:, 1, :, :]
B1 = x_duplicated[:, 2, :, :]
L2 = x_duplicate_interleaved[:, 0, :, :]
A2 = x_duplicate_interleaved[:, 1, :, :]
B2 = x_duplicate_interleaved[:, 2, :, :]

print(f"L1: {L1} -> L2: {L2}")
