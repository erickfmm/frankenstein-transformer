"""Per-head Muon and K2 weight clipping (Kimi K3, arXiv:2607.24653 §2.5)."""

import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.model.optimizer.muon import Muon


def _group(param, name, *, per_head, heads, clip=0.0, lr=0.1):
    return {
        "params": [param],
        "name": "attention",
        "param_names": [name],
        "lr": lr,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "nesterov": False,
        "ns_steps": 5,
        "ns_eps": 1e-7,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "per_head": per_head,
        "per_head_heads": heads,
        "clip_value": clip,
    }


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class PerHeadMuonTests(unittest.TestCase):
    def test_single_head_equals_full_matrix(self):
        torch.manual_seed(0)
        init = torch.randn(8, 4)
        grad = torch.randn(8, 4)
        p_full = torch.nn.Parameter(init.clone())
        p_head = torch.nn.Parameter(init.clone())
        p_full.grad = grad.clone()
        p_head.grad = grad.clone()
        Muon([_group(p_full, "q_proj.weight", per_head=False, heads=0)]).step()
        Muon([_group(p_head, "q_proj.weight", per_head=True, heads=1)]).step()
        torch.testing.assert_close(p_full, p_head)

    def test_per_head_matches_manual_block_orthogonalization(self):
        torch.manual_seed(0)
        init = torch.randn(8, 4)
        grad = torch.randn(8, 4)
        param = torch.nn.Parameter(init.clone())
        param.grad = grad.clone()
        Muon(
            [_group(param, "q_proj.weight", per_head=True, heads=2, lr=0.1)]
        ).step()
        blocks = grad.reshape(2, 4, 4)
        expected_update = torch.stack(
            [Muon._orthogonalize(blocks[h], 5, 1e-7) for h in range(2)]
        ).reshape_as(grad)
        torch.testing.assert_close(
            param, init - 0.1 * expected_update.to(init.dtype)
        )

    def test_clip_bounds_the_post_update_weights(self):
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(8, 4) * 10.0)
        param.grad = torch.randn(8, 4) * 10.0
        Muon(
            [_group(param, "q_proj.weight", per_head=False, heads=0,
                    clip=0.25, lr=1.0)]
        ).step()
        self.assertTrue(bool((param.abs() <= 0.25 + 1e-6).all()))

    def test_non_projection_matrix_is_not_split(self):
        torch.manual_seed(0)
        init = torch.randn(8, 4)
        grad = torch.randn(8, 4)
        p_split = torch.nn.Parameter(init.clone())
        p_full = torch.nn.Parameter(init.clone())
        p_split.grad = grad.clone()
        p_full.grad = grad.clone()
        Muon([_group(p_split, "out_proj.weight", per_head=True, heads=2)]).step()
        Muon([_group(p_full, "out_proj.weight", per_head=False, heads=0)]).step()
        torch.testing.assert_close(p_split, p_full)


if __name__ == "__main__":
    unittest.main()
