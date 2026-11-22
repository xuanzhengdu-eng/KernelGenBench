"""Test inputs for GGML quantized weights."""
import torch


def make_q8_0_data(row_shape, columns, device):
    """Build rows of Q8_0 blocks with an FP16 scale and 32 int8 values."""
    block_shape = (*row_shape, columns // 32)
    scales = (torch.randn(*block_shape, 1).abs() * 0.5 + 0.1).to(torch.float16)
    values = torch.randint(0, 256, (*block_shape, 32), dtype=torch.uint8)
    return torch.cat((scales.view(torch.uint8), values), dim=-1).flatten(-2).to(device)
