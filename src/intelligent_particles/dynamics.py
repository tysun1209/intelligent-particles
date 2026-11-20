import numpy as np

from .state import ParticleState


def advance(state, turn, world, rng):
    """Euler step with fixed propulsion speed and rotational diffusion."""
    noise = np.sqrt(2 * world.rotational_diffusion * world.dt) * rng.normal(size=world.n)
    heading = (state.heading + turn + noise + np.pi) % (2 * np.pi) - np.pi
    velocity = world.speed * np.column_stack((np.cos(heading), np.sin(heading)))
    return ParticleState((state.position + world.dt * velocity) % world.box, heading)
