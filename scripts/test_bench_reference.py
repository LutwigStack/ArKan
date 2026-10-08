"""Small parity gate for the benchmark baselines; run with unittest."""
import json
from pathlib import Path
import unittest

import torch
import bench_pytorch as cpu
import bench_pytorch_train as training
import bench_competitors as competitors
import bench_pytorch_gpu as gpu


class ReferenceParity(unittest.TestCase):
    def test_range_and_fixture_basis(self):
        root = Path(__file__).parent.parent / 'tests/reference_data'
        paths = list(root.glob('training_ref_*.json'))
        self.assertEqual(len(paths), 5, 'expected all five checked-in same-weight fixtures')
        for module in (cpu, training, gpu):
            for path in paths:
                fixture = json.loads(path.read_text())
                c = fixture['config']
                cfg = module.KanConfig(input_dim=c['input_dim'], output_dim=c['output_dim'], hidden_dims=tuple(c['hidden_dims']), grid_size=c['grid_size'], spline_order=c['spline_order'], grid_range=tuple(c['grid_range']))
                knots = module.compute_knots(cfg, torch.device("cpu")) if module is gpu else module.compute_knots(cfg)
                network = []
                for i, layer in enumerate(fixture['init_weights']):
                    network.append({'in': c['dims'][i], 'out': c['dims'][i+1], 'weights': torch.tensor(layer['weights']).reshape(c['dims'][i+1], c['dims'][i], c['global_basis_size']), 'bias': torch.tensor(layer['bias'])})
                x = torch.tensor(fixture['dataset']['inputs']).reshape(-1, cfg.input_dim)
                forward = module.forward_arkan_style_gpu if module is gpu else module.forward_vectorized
                actual = forward(network, cfg, x, knots).flatten()
                torch.testing.assert_close(actual, torch.tensor(fixture['step0_forward']), atol=1e-5, rtol=1e-5)

    def test_competitor_matches_existing_multilayer_fixture(self):
        fixture = json.loads((Path(__file__).parent.parent / 'tests/reference_data/training_ref_multilayer.json').read_text())
        c = fixture['config']
        model = competitors.FaithfulKAN(c['dims'], c['grid_size'], c['spline_order'])
        with torch.no_grad():
            for layer, reference in zip(model.kan_layers, fixture['init_weights']):
                layer.weights.copy_(torch.tensor(reference['weights']).reshape_as(layer.weights))
                layer.bias.copy_(torch.tensor(reference['bias']))
        x = torch.tensor(fixture['dataset']['inputs']).reshape(-1, c['input_dim'])
        knots = competitors.compute_knots(c['grid_size'], c['spline_order'], c['grid_range'])
        torch.testing.assert_close(model(x, knots).flatten(), torch.tensor(fixture['step0_forward']), atol=1e-5, rtol=1e-5)

    def test_competitor_global_coefficient_gather_and_clamp(self):
        layer = competitors.FaithfulKANLayer(1, 1, 5, 3)
        self.assertEqual(tuple(layer.weights.shape), (1, 1, 8))
        with torch.no_grad():
            layer.weights.copy_(torch.arange(8).reshape(1, 1, 8))
        knots = competitors.compute_knots(5, 3)
        x = torch.tensor([[-30.0], [-3.0], [0.0], [3.0], [30.0]])
        result = layer(x, knots).flatten()
        torch.testing.assert_close(result, torch.tensor([1.0, 1.0, 3.5, 6.0, 6.0]))

    def test_narrow_grid_preserves_linear_spline(self):
        cfg = cpu.KanConfig(input_dim=1, output_dim=1, hidden_dims=(), grid_range=(-1e-7, 1e-7))
        network = [{'in': 1, 'out': 1, 'weights': torch.arange(8, dtype=torch.float32).reshape(1, 1, 8), 'bias': torch.zeros(1)}]
        x = torch.tensor([[-1e-7], [0.0], [1e-7]])
        result = cpu.forward_vectorized(network, cfg, x, cpu.compute_knots(cfg)).flatten()
        torch.testing.assert_close(result, torch.tensor([1.0, 3.5, 6.0]))

    def test_reset_repeats_same_adam_step(self):
        from bench_reference import capture_reset
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        reset = capture_reset(model, optimizer)
        inputs = torch.tensor([[0.2, 0.7]])
        observations = []
        for _ in range(3):
            reset()
            loss = model(inputs).square().sum()
            loss.backward()
            optimizer.step()
            observations.append((loss.detach().clone(), model.weight.detach().clone(), optimizer.state[model.weight]['step'].clone()))
        for actual in observations[1:]:
            for a, b in zip(actual, observations[0]):
                torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)

    def test_training_autograd_matches_linear_spline(self):
        cfg = training.KanConfig(input_dim=1, output_dim=1, hidden_dims=())
        weights = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8).requires_grad_()
        x = torch.tensor([[-10.0], [0.0], [10.0]], requires_grad=True)
        network = [{'in': 1, 'out': 1, 'weights': weights, 'bias': torch.zeros(1, requires_grad=True)}]
        result = training.forward_vectorized(network, cfg, x, training.compute_knots(cfg))
        result.sum().backward()
        torch.testing.assert_close(x.grad.flatten(), torch.tensor([0.0, 5.0 / 6.0, 0.0]))
        self.assertAlmostEqual(weights.grad.sum().item(), 3.0, places=5)


if __name__ == '__main__':
    unittest.main()
