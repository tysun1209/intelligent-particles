import importlib.util
import unittest

import numpy as np

from intelligent_particles.replay import Replay


class ReplayTest(unittest.TestCase):
    def test_oversized_batch_keeps_latest_particles(self):
        buffer = Replay(3, 1)
        x = np.arange(5, dtype=float)
        buffer.add(x[:, None], x, -x, x[:, None] + 1, x == 4)
        states, actions, rewards, following, done = buffer.fields
        np.testing.assert_array_equal(states[:, 0], [2, 3, 4])
        np.testing.assert_array_equal(actions, states)
        np.testing.assert_array_equal(rewards, -states)
        np.testing.assert_array_equal(following, states + 1)
        np.testing.assert_array_equal(done[:, 0], [0, 0, 1])


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires learning extra")
class SteeringTest(unittest.TestCase):
    def test_updates_actor_and_outputs_one_bounded_turn(self):
        import torch
        from intelligent_particles import PeriodicParticles
        from intelligent_particles.learning import train_turn_sac
        from intelligent_particles.reward import eastward_reward
        from intelligent_particles.models.sac import SAC

        torch.manual_seed(12)
        initial = SAC(4, 16, "cpu")
        world = PeriodicParticles(n=8, reward=eastward_reward)
        agent, returns = train_turn_sac(world, episodes=2, steps=3, seed=12,
                                        hidden=16, batch_size=8, capacity=32)
        self.assertTrue(np.isfinite(returns).all())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial.actor.parameters(), agent.actor.parameters())))
        turn = agent.normalized_turn(world.reset(17), deterministic=True)
        self.assertEqual(turn.shape, (8,))
        self.assertTrue(np.all(np.abs(turn) <= 1))
