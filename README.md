# Differentiable Optimal Transport (DOT)

PyTorch operators for differentiable optimal transport with CUDA acceleration.

## Installation

```bash
pip install -e .
```

Or build manually:

```bash
meson setup builddir
meson compile -C builddir
pip install -e .
```

## Usage

```python
import torch
import dot

# Create cost matrix
cost = torch.rand(batch_size, M, N, device='cuda')

# Compute optimal transport
result = dot.sinkhorn(cost, reg=1.0)

# Access transport plan and cost
transport_plan = result.transport_plan  # (B, M, N)
transport_cost = result.cost  # (B,)

# Fully differentiable
loss = transport_cost.sum()
loss.backward()
```

### Module API

```python
sinkhorn = dot.Sinkhorn(reg=1.0, max_iter=100)
result = sinkhorn(cost_matrix)
```

## Development

Run tests:

```bash
pytest tests/
```

## License

MIT
