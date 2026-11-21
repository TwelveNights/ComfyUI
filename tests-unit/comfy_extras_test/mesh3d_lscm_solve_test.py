"""The batched LSCM solve splits batches that exhaust hipBLAS's LU workspace."""

import numpy as np
import pytest
import torch

from comfy.cli_args import args as cli_args

# Model management probes the device on import, before pytest fixtures run.
_original_cpu = cli_args.cpu
try:
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        cli_args.cpu = True
    from comfy_extras.mesh3d.uv_unwrap import parameterize
finally:
    cli_args.cpu = _original_cpu

ALLOC_FAILED = ("CUDA error: HIPBLAS_STATUS_ALLOC_FAILED when calling `hipblasDgetrfBatched( handle, n, "
                "dA_array, ldda, ipiv_array, info_array, batchsize)`")


def _systems(batch, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(batch, n, n, generator=g, dtype=torch.float64)
    return a @ a.transpose(1, 2) + n * torch.eye(n, dtype=torch.float64), torch.randn(batch, n, generator=g, dtype=torch.float64)


def _workspace_limited_solve(monkeypatch, max_batch, fail_always=False):
    """torch.linalg.solve that fails like hipBLAS for batches above `max_batch`."""
    solve = torch.linalg.solve
    calls = []

    def limited(a, b):
        calls.append(a.shape[0])
        if fail_always or a.shape[0] > max_batch:
            raise RuntimeError(ALLOC_FAILED)
        return solve(a, b)

    monkeypatch.setattr(torch.linalg, "solve", limited)
    return calls


def test_solves_directly_when_the_workspace_suffices(monkeypatch):
    a, b = _systems(8)
    calls = _workspace_limited_solve(monkeypatch, max_batch=8)
    x = parameterize._solve_batched_gpu(a, b)
    torch.testing.assert_close(x, torch.from_numpy(np.linalg.solve(a.numpy(), b.numpy()[..., None])[..., 0]))
    assert calls == [8]


def test_splits_batches_that_exhaust_the_workspace(monkeypatch):
    a, b = _systems(43)
    calls = _workspace_limited_solve(monkeypatch, max_batch=10)
    x = parameterize._solve_batched_gpu(a, b)
    torch.testing.assert_close(x, torch.from_numpy(np.linalg.solve(a.numpy(), b.numpy()[..., None])[..., 0]))
    # every system is solved exactly once, on the GPU path, in batches that fit
    assert calls[0] == 43
    assert sum(c for c in calls if c <= 10) == 43


def test_single_system_falls_back_to_cpu(monkeypatch):
    a, b = _systems(3)
    _workspace_limited_solve(monkeypatch, max_batch=0, fail_always=True)
    x = parameterize._solve_batched_gpu(a, b)
    torch.testing.assert_close(x, torch.from_numpy(np.linalg.solve(a.numpy(), b.numpy()[..., None])[..., 0]))


@pytest.mark.parametrize("message", [
    "linalg.solve: The solver failed because the input matrix is singular.",
    "CUDA error: an illegal memory access was encountered",
])
def test_other_errors_propagate(monkeypatch, message):
    a, b = _systems(4)

    def fail(a, b):
        raise RuntimeError(message)

    monkeypatch.setattr(torch.linalg, "solve", fail)
    with pytest.raises(RuntimeError, match=message.split(":")[0]):
        parameterize._solve_batched_gpu(a, b)
