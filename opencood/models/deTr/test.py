

import torch

from opencood.models.deTr.DeAttn import MSDeformAttn
from opencood.models.deTr.MSDAFunc import MSDeformAttnFunction

# Parameters
batch_size = 2
num_heads = 2
embed_dim = 4
num_levels = 3
num_points = 4

# Input feature map (value), concatenating all levels into a single tensor
# Shape: (batch_size, total_num_queries, num_heads * embed_dim)
total_num_queries = sum([6 * 6, 4 * 4, 2 * 2])  # Total queries across all levels
value = torch.randn(batch_size, total_num_queries, num_heads , embed_dim)

# input_spatial_shapes (num_levels, 2): heights and widths of each level
input_spatial_shapes = torch.tensor([[6, 6], [4, 4], [2, 2]])

# input_level_start_index (num_levels,): Start indices for each level in `value`
input_level_start_index = torch.tensor([0, 6 * 6, 6 * 6 + 4 * 4])

# sampling_locations (batch_size, num_queries, num_heads, num_levels, num_points, 2)
sampling_locations = torch.rand(batch_size, total_num_queries, num_heads, num_levels, num_points, 2)

# attention_weights (batch_size, num_queries, num_heads, num_levels, num_points)
attention_weights = torch.rand(batch_size, total_num_queries, num_heads, num_levels, num_points)

# # Test the function
# output = MSDeformAttn.ms_deform_attn_core_pytorch(
#      value, input_spatial_shapes, sampling_locations, attention_weights
# )

msda = MSDeformAttn(embed_dim, num_levels, num_heads, num_points)

##
query = torch.randn(batch_size, total_num_queries, embed_dim)
reference_points = torch.randn(batch_size, total_num_queries, num_levels, 2)
input_flatten = torch.randn(batch_size, total_num_queries, embed_dim)
input_spatial_shapes = torch.tensor([[6, 6], [4, 4], [2, 2]])
input_level_start_index = torch.tensor([0, 6 * 6, 6 * 6 + 4 * 4])
output = msda(query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index )

print("Output shape:", output.shape)
# print("Output values:", output)
