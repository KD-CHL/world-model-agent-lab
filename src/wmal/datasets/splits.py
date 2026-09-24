"""Stable episode-level train, validation, and test assignment."""
import random


def split_episodes(episode_ids, *, seed=0, fractions=(0.8, 0.1, 0.1)):
    ids = list(episode_ids)
    if len(ids) != len(set(ids)) or len(ids) < 5 or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("At least five unique nonnegative episode IDs are required")
    if len(fractions) != 3 or any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must be positive and sum to one")
    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, int(len(ids) * fractions[0]))
    validation_count = max(1, int(len(ids) * fractions[1]))
    if train_count + validation_count >= len(ids):
        train_count = len(ids) - 2
        validation_count = 1
    assignment = {episode: "train" for episode in shuffled[:train_count]}
    assignment.update({episode: "validation" for episode in shuffled[train_count:train_count + validation_count]})
    assignment.update({episode: "test" for episode in shuffled[train_count + validation_count:]})
    return {episode: assignment[episode] for episode in sorted(assignment)}
