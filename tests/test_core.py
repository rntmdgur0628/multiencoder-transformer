import math
import unittest
from unittest import mock
from dataclasses import replace

import torch

from msr import ModelConfig, RestorationModel
from msr.audio import ReferenceSTFT
from msr.attention import local_attention
from msr.position import LieRE, angular_coordinates

torch.set_num_threads(2)


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.cfg = ModelConfig.tiny()

    def test_reference_roundtrip_boundaries(self):
        codec = ReferenceSTFT(64, 16)
        for length in (1, 15, 16, 17, 63, 64, 65, 257):
            x = torch.randn(2, 2, length)
            torch.testing.assert_close(codec.synthesis(codec.analysis(x), length), x, atol=1e-6, rtol=1e-5)

    def test_orthogonal_periodic_and_gradient(self):
        pe = LieRE(2, 8, 4).double()
        theta = torch.tensor([0.2, 1.3], dtype=torch.double)
        p = torch.stack((torch.ones_like(theta), theta.cos(), theta.sin()), -1)
        p2 = torch.stack((torch.ones_like(theta), (theta + 2 * math.pi).cos(), (theta + 2 * math.pi).sin()), -1)
        r = pe.rotations(p)
        eye = torch.eye(4, dtype=torch.double).expand_as(r)
        torch.testing.assert_close(r.transpose(-1, -2) @ r, eye)
        torch.testing.assert_close(r, pe.rotations(p2))
        r[..., 0, 1].sum().backward()
        self.assertGreater(pe.raw.grad.norm().item(), 0)
        self.assertTrue(torch.isfinite(pe.raw.grad).all())

    def test_temporal_has_no_frequency(self):
        model = RestorationModel(self.cfg, "A2")
        branches = model.encode(torch.randn(1, 2, 128))
        self.assertEqual(branches[0].values.shape[2], 1)
        self.assertEqual(branches[0].coordinates[..., 1:].count_nonzero(), 0)
        torch.testing.assert_close(branches[1].coordinates, branches[2].coordinates)

    def test_zero_gate_exactly_recovers_baseline(self):
        models = [RestorationModel(self.cfg, variant).eval() for variant in ("A0", "A1", "A2")]
        x = torch.randn(1, 2, 128)
        baseline = models[0](x)
        for model in models[1:]:
            with torch.no_grad():
                model.injection.gates.zero_()
            torch.testing.assert_close(model(x), baseline, rtol=0, atol=0)

    def test_frozen_adapter_keeps_backbone_and_has_pe_gradient(self):
        model = RestorationModel(self.cfg, "A2")
        model.configure_training(True)
        before = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("injection.")}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.002)
        model(torch.randn(1, 2, 128)).square().mean().backward()
        self.assertGreater(model.injection.position.raw.grad.abs().max().item(), 0)
        optimizer.step()
        for k, value in before.items():
            torch.testing.assert_close(model.state_dict()[k], value, atol=0, rtol=0)

    def test_future_perturbation_tokens_and_delayed_waveform(self):
        for variant in ("A0", "A1", "A2"):
            model = RestorationModel(self.cfg, variant).eval()
            x = torch.randn(1, 2, 320)
            changed = x.clone()
            changed[..., 192:] = torch.randn_like(changed[..., 192:]) * 10
            with torch.no_grad():
                a, b = model(x), model(changed)
            torch.testing.assert_close(a[..., :192 - model.latency_samples], b[..., :192 - model.latency_samples], atol=2e-6, rtol=1e-5)

    def test_kv_joint_permutation(self):
        model = RestorationModel(self.cfg, "A2").eval()
        branches = model.encode(torch.randn(1, 2, 128))
        memories = []
        for branch in branches:
            perm = torch.randperm(branch.values.shape[2])
            memories.append(replace(branch, values=branch.values[:, :, perm],
                                    coordinates=branch.coordinates[:, perm], valid=branch.valid[:, :, perm]))
        original, _ = model.injection(branches, True)
        permuted, _ = model.injection(branches, True, memories=memories)
        for a, b in zip(original, permuted):
            torch.testing.assert_close(a.values, b.values, atol=1e-7, rtol=1e-6)

    def test_all_masked_is_zero_and_finite(self):
        q = torch.randn(1, 3, 2, 2, 4, requires_grad=True)
        valid = torch.zeros(1, 3, 2, dtype=torch.bool)
        y = local_attention(q, q, q, valid, valid, 3)
        self.assertEqual(y.count_nonzero(), 0)
        y.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_same_hz_same_coordinate_across_grids(self):
        def coords(f):
            return angular_coordinates(torch.tensor([0.0]), torch.tensor(f), torch.ones(len(f), dtype=torch.bool), 48000, math.pi)
        torch.testing.assert_close(coords([1000.0, 2000.0])[:, 0], coords([1000.0, 3000.0, 5000.0])[:, 0])

    def test_backward_all_arms_and_silence(self):
        for variant in ("A0", "A1", "A2"):
            model = RestorationModel(self.cfg, variant)
            y = model(torch.zeros(1, 2, 65))
            self.assertEqual(tuple(y.shape), (1, 6, 2, 65))
            self.assertTrue(torch.isfinite(y).all())
            y.square().sum().backward()

    def test_bad_release_clock_is_rejected(self):
        model = RestorationModel(self.cfg, "A2")
        branches = model.encode(torch.randn(1, 2, 128))
        bad = [replace(b, release=b.release.flip(0)) for b in branches]
        with self.assertRaisesRegex(ValueError, "release samples"):
            model.injection(bad, True)

    def test_coordinates_only_shuffle_changes_message(self):
        model = RestorationModel(self.cfg, "A2").eval()
        branches = model.encode(torch.randn(1, 2, 128))
        memory = [replace(b, coordinates=b.coordinates.flip(1)) for b in branches]
        a, _ = model.injection(branches, True)
        b, _ = model.injection(branches, True, memories=memory)
        self.assertGreater(max((x.values - y.values).abs().max().item() for x, y in zip(a, b)), 0)

    def test_late_change_in_one_branch_cannot_affect_earlier_message(self):
        model = RestorationModel(self.cfg, "A2")
        branches = model.encode(torch.randn(1, 2, 128))
        memory = [replace(b, values=b.values.clone()) for b in branches]
        memory[2].values[:, 4:] += 100
        a, _ = model.injection(branches, True)
        b, _ = model.injection(branches, True, memories=memory)
        for x, y in zip(a, b):
            torch.testing.assert_close(x.values[:, :4], y.values[:, :4], rtol=0, atol=0)

    def test_end_to_end_joint_permutation_and_wrong_track(self):
        model = RestorationModel(self.cfg, "A2").eval()
        x, donor = torch.randn(1, 2, 128), torch.randn(1, 2, 128)
        with torch.no_grad():
            y = model(x)
            torch.testing.assert_close(y, model(x, intervention="kv_joint_permutation"), atol=2e-6, rtol=1e-5)
            shuffled = model(x, intervention="frequency_shuffle")
            wrong = model(x, intervention="wrong_track", memory_waveform=donor)
        self.assertGreater((y - shuffled).abs().max().item(), 0)
        self.assertGreater((y - wrong).abs().max().item(), 0)

    def test_small_one_batch_fit_reduces_loss(self):
        model = RestorationModel(self.cfg, "A0")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        x = torch.randn(1, 2, 128) * 0.1
        y = x[:, None].expand(-1, 6, -1, -1) / 6
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * 0.7)

    def test_local_attention_calls_four_dimensional_sdpa(self):
        original = torch.nn.functional.scaled_dot_product_attention
        def checked(q, k, v, **kwargs):
            self.assertEqual(q.ndim, 4)
            self.assertEqual(k.ndim, 4)
            return original(q, k, v, **kwargs)
        q = torch.randn(2, 3, 4, 2, 8)
        valid = torch.ones(2, 3, 4, dtype=torch.bool)
        with mock.patch("msr.attention.F.scaled_dot_product_attention", side_effect=checked):
            y = local_attention(q, q, q, valid, valid, 16)
        self.assertEqual(y.shape, q.shape)


if __name__ == "__main__":
    unittest.main()
