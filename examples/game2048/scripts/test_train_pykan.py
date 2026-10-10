"""Exercise the actual target-building code without optional Torch/KAN imports.

NumPy executes the scalar fixture arithmetic; Torch training is a separate check.
"""
import ast
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np


class Array(np.ndarray):
    def to(self, _device):
        return self


def target_builder(source=None):
    source = source or Path(__file__).with_name("train_pykan.py")
    tree = ast.parse(Path(source).read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in ("Game2048", "legal_next_q")]
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DQNAgent")
    method = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == "train_batch")
    target_block = next(node for node in method.body if isinstance(node, ast.With))
    namespace = dict(np=np, random=random, device="cpu", torch=SimpleNamespace(
        FloatTensor=lambda values: np.asarray(values, dtype=np.float32).view(Array)))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, target_block.body


class TargetTests(unittest.TestCase):
    def target(self, board, q_values, reward=0.0, done=False):
        namespace, body = target_builder()
        game = namespace["Game2048"].__new__(namespace["Game2048"])
        game.board = np.asarray(board, dtype=np.int32).reshape(4, 4)
        state = game.get_state()
        namespace.update(self=SimpleNamespace(gamma=1.0, target_net=lambda _: np.array([q_values])),
                         next_states=[state], next_states_t=np.array([state]), dones=[done],
                         rewards_t=np.array([reward]), dones_t=np.array([float(done)]))
        exec(compile(ast.Module(body=body, type_ignores=[]), "target_block", "exec"), namespace)
        return float(namespace["target_q"][0])

    def test_invalid_up_does_not_bootstrap(self):
        # Python directions are Left, Right, Up, Down; Up is board-preserving here.
        board = [2, 0, 0, 4] + [0] * 12
        self.assertEqual(self.target(board, [3.0, 1.0, 100.0, 2.0]), 3.0)

    def test_terminal_and_no_legal_action_are_reward_only(self):
        board = [2, 0, 0, 4] + [0] * 12
        self.assertEqual(self.target(board, [3.0, 1.0, 100.0, 2.0], 2.0, True), 2.0)
        self.assertEqual(self.target(board, [float("nan")] * 4, 2.0, True), 2.0)
        self.assertEqual(self.target([0] * 16, [100.0] * 4, 2.0), 2.0)


if __name__ == "__main__":
    unittest.main()
