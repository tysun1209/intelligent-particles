"""Compare local alignment with independent rotational diffusion from paired initial states."""
from intelligent_particles import PeriodicParticles, align_with_neighbors, rollout

from intelligent_particles.models.independent import independent
from intelligent_particles.reward import alignment_reward


if __name__ == "__main__":
    for policy in (independent, align_with_neighbors):
        world = PeriodicParticles(n=48, reward=alignment_reward)
        for transition in rollout(world, policy, seed=42, steps=100):
            pass
        print(f"{policy.__name__}: final mean alignment reward {transition.reward.mean():.3f}")
