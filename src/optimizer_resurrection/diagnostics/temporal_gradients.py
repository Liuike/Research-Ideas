from __future__ import annotations

import torch


@torch.no_grad()
def temporal_jacobian_statistics(
    recurrent_weight: torch.Tensor,
    states: torch.Tensor,
    lengths: torch.Tensor,
    lags: tuple[int, ...] = (1, 5, 10, 20, 50, 100),
    max_examples: int = 8,
) -> dict[str, dict[str, float | list[float]]]:
    """Estimate each sequence's J_(L-k -> L), excluding padded timesteps."""
    result: dict[str, dict[str, float | list[float]]] = {}
    for lag in lags:
        eligible = torch.nonzero(lengths >= lag, as_tuple=False).flatten()[:max_examples]
        if eligible.numel() == 0:
            continue
        spectra = []
        frobenius = []
        for index in eligible.tolist():
            end = int(lengths[index])
            product = torch.eye(recurrent_weight.shape[0], device=states.device)
            for t in range(end - lag, end):
                derivative = 1 - states[index, t].square()
                product = torch.diag(derivative) @ recurrent_weight @ product
            spectra.append(torch.linalg.svdvals(product.float()))
            frobenius.append(product.norm())
        values = torch.stack(spectra)
        all_values = values.flatten()
        quantiles = torch.quantile(all_values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
        result[str(lag)] = {
            "spectral_norm": float(values[:, 0].mean()),
            "frobenius_norm": float(torch.stack(frobenius).mean()),
            "singular_quantiles": [float(v) for v in quantiles],
            "examples": int(eligible.numel()),
        }
    return result
