"""Compare independent or locally aligned periodic particles."""
import argparse
import sys

import numpy as np

from .environment import PeriodicParticles
from .models.alignment import align_with_neighbors
from .models.independent import independent
from .reward import alignment_reward
from .runner import rollout


def run_periodic(argv):
    parser = argparse.ArgumentParser(prog="intelligent_particles periodic")
    parser.add_argument("--model", choices=("alignment", "independent"), default="alignment")
    parser.add_argument("--num-agents", type=int, default=48)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.num_agents, args.steps) < 1 or args.seed < 0:
        parser.error("population and steps must be positive and seed must be nonnegative")
    model = {"alignment": align_with_neighbors, "independent": independent}[args.model]
    world = PeriodicParticles(n=args.num_agents, reward=alignment_reward)
    for step in rollout(world, model, seed=args.seed, steps=args.steps):
        final_alignment = np.mean(step.reward)
    print(f"{args.model}: final mean alignment reward {final_alignment:.3f}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("periodic",))
    parser.parse_args(argv[:1])
    run_periodic(argv[1:])


if __name__ == "__main__":
    main()
