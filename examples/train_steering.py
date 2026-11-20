"""Short SAC training example on a periodic domain with a custom reward."""
from intelligent_particles import PeriodicParticles, rollout
from intelligent_particles.learning import train_turn_sac
from intelligent_particles.reward import eastward_reward


if __name__ == "__main__":
    world = PeriodicParticles(n=16, reward=eastward_reward)
    agent, returns = train_turn_sac(world, episodes=3, steps=10, seed=7,
                                    max_turn=0.3, hidden=32, batch_size=32, capacity=2048)
    def policy(obs, rng):
        return 0.3 * agent.get_deterministic_action_batch(obs)[:, 0]
    score = sum(tr.reward.mean() for tr in rollout(world, policy, seed=109, steps=10))
    print(f"Training episodes: {len(returns)}; evaluation return per particle: {score:.3f}")
