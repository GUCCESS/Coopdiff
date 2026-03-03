import torch
import torch.nn.functional as F


class MSDeformAttnFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
                im2col_step=64):
        """
        Simplified Python implementation of Deformable Attention.
        Args:
            value: Input features of shape (batch_size, total_num_queries, num_heads * embed_dim)
            input_spatial_shapes: Tensor of shape (num_levels, 2), containing (height, width) of each feature level
            input_level_start_index: Tensor of shape (num_levels,), containing start index of each feature level
            sampling_locations: Sampling locations of shape (batch_size, num_queries, num_heads, num_levels, num_points, 2)
            attention_weights: Attention weights of shape (batch_size, num_queries, num_heads, num_levels, num_points)
        Returns:
            output: Aggregated output features of shape (batch_size, num_queries, embed_dim)
        """
        batch_size, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
        embed_dim = value.shape[-1] // num_heads  # Assuming value has shape (batch_size, total_num_queries, embed_dim)

        # Reshape the value tensor for processing
        value = value.view(batch_size, -1, num_heads, embed_dim)

        # Placeholder for output aggregation
        output = torch.zeros(batch_size, num_queries, num_heads, embed_dim, device=value.device)

        # Iterate over each feature level (different resolution scales)
        for lvl in range(num_levels):
            # Extract the corresponding feature level dimensions
            H_, W_ = input_spatial_shapes[lvl]
            level_value = value[:, input_level_start_index[lvl]:input_level_start_index[lvl] + H_ * W_]
            level_value = level_value.view(batch_size, H_, W_, num_heads, embed_dim)  # Reshape to (B, H, W, heads, dim)

            # Sampling locations for this level, shape (B, num_queries, num_heads, num_points, 2)
            sampling_loc = sampling_locations[:, :, :, lvl]

            print(sampling_loc.shape)

            # Rescale sampling locations to match the current feature level size (H_, W_)
            sampling_loc = sampling_loc * torch.tensor([W_, H_], device=value.device).view(1, 1, 1, 1, 2)

            print(sampling_loc.shape)
            # Perform bilinear interpolation at the sampled locations
            sampled_value = MSDeformAttnFunction.bilinear_interpolate(level_value, sampling_loc)

            # Apply the attention weights to the sampled values
            attn_weight = attention_weights[:, :, :, lvl].view(batch_size, num_queries, num_heads, num_points, 1)
            weighted_value = (sampled_value * attn_weight).sum(dim=-2)  # Sum over num_points

            # Aggregate results for this level
            output += weighted_value

        # Reshape the output back to (batch_size, num_queries, embed_dim)
        output = output.view(batch_size, num_queries, -1)
        return output

    @staticmethod
    def bilinear_interpolate(input, coords):
        """
        Bilinear interpolation over the input feature map at given coordinates.
        Args:
            input: Input feature map of shape (batch_size, height, width, num_heads, embed_dim)
            coords: Sampling locations of shape (batch_size, num_queries, num_heads, num_points, 2)
        Returns:
            interpolated_values: Interpolated feature values at the sampling locations
        """


        print(input.shape)
        print(coords.shape)
        batch_size, H, W, num_heads, embed_dim = input.shape
        x = coords[..., 0]  # (batch_size, num_queries, num_heads, num_points)
        y = coords[..., 1]  # (batch_size, num_queries, num_heads, num_points)

        # Clamp the coordinates to be within the bounds of the image
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)

        # Compute the coordinates of the four corners
        x0 = x.floor().long()
        x1 = x0 + 1
        y0 = y.floor().long()
        y1 = y0 + 1

        # Clip the coordinates to the image boundaries
        x1 = x1.clamp(0, W - 1)
        y1 = y1.clamp(0, H - 1)

        # Gather the pixel values at the four corners
        Ia = input[:, y0, x0]  # top-left
        Ib = input[:, y1, x0]  # bottom-left
        Ic = input[:, y0, x1]  # top-right
        Id = input[:, y1, x1]  # bottom-right

        # Compute the bilinear weights
        wa = (x1.float() - x) * (y1.float() - y)
        wb = (x1.float() - x) * (y - y0.float())
        wc = (x - x0.float()) * (y1.float() - y)
        wd = (x - x0.float()) * (y - y0.float())

        # Compute the interpolated feature values
        interpolated_values = (wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id)
        return interpolated_values



