import unittest
import numpy as np
from intelligent_particles import ParticleState, PeriodicParticles, Step, align_with_neighbors, local_headings, rollout


class CoreTest(unittest.TestCase):
    def test_reproducible_and_swappable_reward(self):
        task = PeriodicParticles(n=8, reward=lambda prev, action, current, world: np.ones(world.n))
        def run():
            return list(rollout(task, align_with_neighbors, seed=31, steps=4))
        a, b = run(), run()
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.next_observation, y.next_observation)
            np.testing.assert_array_equal(x.reward, np.ones(8))
        self.assertTrue(a[-1].truncated)
        self.assertFalse(a[0].truncated)
        self.assertFalse(np.shares_memory(a[0].next_observation, a[1].observation))

    def test_custom_action_is_passed_to_dynamics_and_reward(self):
        from intelligent_particles.action import turn_angles
        from intelligent_particles.periodic import PeriodicParticles as LegacyWorld

        def degrees_to_radians(action, world):
            return np.deg2rad(turn_angles(action, world))

        world = LegacyWorld(n=2, rotational_diffusion=0, action=degrees_to_radians,
                            reward=lambda previous, turn, current, world: turn.copy())
        world.reset(11)
        heading = world.state.heading.copy()
        step = world.step([90, -90])
        np.testing.assert_allclose(step.reward, [np.pi / 2, -np.pi / 2])
        np.testing.assert_allclose(world.state.heading, (heading + step.reward + np.pi) % (2 * np.pi) - np.pi)
        self.assertIs(LegacyWorld, PeriodicParticles)

    def test_periodic_neighbors(self):
        state = ParticleState(np.array([[.1, 2], [9.9, 2], [5, 5]]), np.array([0., np.pi/2, 1.]))
        obs = local_headings(state, PeriodicParticles(n=3, box=10, radius=.5))
        np.testing.assert_allclose(align_with_neighbors(obs, None), [np.pi/2, -np.pi/2, 0])

    def test_per_particle_termination(self):
        class World:
            def reset(self, seed):
                self.i = 0
                return np.zeros((2, 1))
            def step(self, action):
                self.i += 1
                return Step(np.zeros((2, 1)), np.ones(2), np.array([True, self.i == 2]))
        values = list(rollout(World(), lambda obs, rng: np.zeros(2), seed=8, steps=4))
        self.assertEqual(len(values), 2)
        np.testing.assert_equal(values[1].active, [False, True])
        self.assertFalse(values[-1].truncated)

