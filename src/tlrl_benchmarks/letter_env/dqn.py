"""DQN machinery for the frozen DeepLTL LetterEnv benchmark.

The module is importable without DeepLTL or PyTorch installed. Runtime pieces
that need those dependencies import them lazily, so the semantic test suite can
still run in the light v3 environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Iterable, Sequence


EPSILON_ACTION = -42


@dataclass(frozen=True, slots=True)
class DQNConfig:
    """Frozen DQN hyperparameters for LetterEnv training."""

    total_steps: int = 5_000_000
    num_envs: int = 8
    buffer_size: int = 200_000
    learning_starts: int = 10_000
    batch_size: int = 128
    train_frequency: int = 4
    target_update_frequency: int = 10_000
    checkpoint_interval: int = 250_000
    log_interval: int = 10_000
    discount: float = 0.94
    learning_rate: float = 1e-4
    adam_eps: float = 1e-5
    max_grad_norm: float = 10.0
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    epsilon_decay_steps: int = 1_000_000
    double_dqn: bool = True
    seed: int = 930001
    name: str = "letter_env_dqn_v2_cached_curriculum_recency_fix"
    device: str = "cuda"
    num_threads: int | None = 8
    replay_preprocessing: str = "cached"

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")
        if self.train_frequency <= 0:
            raise ValueError("train_frequency must be positive")
        if self.target_update_frequency <= 0:
            raise ValueError("target_update_frequency must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.adam_eps <= 0 or not math.isfinite(self.adam_eps):
            raise ValueError("adam_eps must be finite and positive")
        if self.max_grad_norm <= 0 or not math.isfinite(self.max_grad_norm):
            raise ValueError("max_grad_norm must be finite and positive")
        if not 0 <= self.final_epsilon <= self.initial_epsilon <= 1:
            raise ValueError("epsilon bounds must satisfy 0 <= final <= initial <= 1")
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive")
        if not self.name:
            raise ValueError("name cannot be empty")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, auto")
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive when set")
        if self.replay_preprocessing not in {"cached", "upstream"}:
            raise ValueError("replay_preprocessing must be one of: cached, upstream")

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Any
    action: int
    reward: float
    next_observation: Any
    done: bool

    def __post_init__(self) -> None:
        if not isinstance(self.action, int):
            raise TypeError("action must be an integer")
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "done", bool(self.done))


@dataclass(frozen=True, slots=True)
class CachedObservation:
    """Compact, device-independent result of one observation encoding.

    DeepLTL's stock ``preprocess_obss`` converts every logical assignment to
    vocabulary ids and then performs many tiny CPU-to-GPU copies every time a
    replay transition is sampled. Replay reuses transitions, so that invariant
    work belongs at collection time. Features stay in their compact environment
    dtype (uint8 for LetterEnv) until a sampled batch is moved to the model.
    """

    features: Any
    reach: tuple[tuple[int, ...], ...]
    avoid: tuple[tuple[int, ...], ...]
    epsilon_enabled: bool


def encode_observation(observation: dict[str, Any], propositions: set[str]) -> CachedObservation:
    """Encode one nonterminal DeepLTL observation exactly once on the CPU."""

    import numpy as np
    from ltl.automata import LDBASequence
    from ltl.logic import Assignment
    from preprocessing.preprocessing import preprocess_sequence

    sequence = list(reversed(observation["goal"]))
    if not sequence:
        raise ValueError("cannot encode an observation with an empty remaining goal")
    encoded_sequence = preprocess_sequence(sequence)
    reach = tuple(tuple(int(token) for token in item[0]) for item in encoded_sequence)
    avoid = tuple(tuple(int(token) for token in item[1]) for item in encoded_sequence)

    epsilon_enabled = sequence[-1][0] == LDBASequence.EPSILON
    if epsilon_enabled and len(sequence) > 1:
        next_avoid = sequence[-2][1]
        assignment = Assignment(
            {proposition: proposition in observation["propositions"] for proposition in propositions}
        ).to_frozen()
        epsilon_enabled = assignment not in next_avoid

    features = np.asarray(observation["features"]).copy()
    features.setflags(write=False)
    return CachedObservation(
        features=features,
        reach=reach,
        avoid=avoid,
        epsilon_enabled=bool(epsilon_enabled),
    )


def _pad_token_sequences(
    sequences: Sequence[tuple[tuple[int, ...], ...]],
    *,
    device: Any,
):
    """Batch cached token ids with one host allocation and one device copy."""

    import numpy as np
    import torch

    lengths = np.fromiter((len(sequence) for sequence in sequences), dtype=np.int64)
    if not len(lengths) or int(lengths.min()) <= 0:
        raise ValueError("cached logical sequences must be nonempty")
    max_length = int(lengths.max())
    max_set_size = max(len(tokens) for sequence in sequences for tokens in sequence)
    data = np.zeros((len(sequences), max_length, max_set_size), dtype=np.int64)
    for batch_index, sequence in enumerate(sequences):
        for sequence_index, tokens in enumerate(sequence):
            data[batch_index, sequence_index, : len(tokens)] = tokens
    # pack_padded_sequence requires lengths on CPU. Token data is copied to
    # the model device once per component, instead of once per logical set.
    return torch.from_numpy(lengths), torch.from_numpy(data).to(device=device)


def collate_cached_observations(
    observations: Sequence[CachedObservation],
    *,
    device: Any,
):
    """Collate cached observations into the exact input contract of LTLNet."""

    if not observations:
        raise ValueError("cannot collate an empty observation batch")

    import numpy as np
    import torch
    import torch_ac

    feature_array = np.stack([observation.features for observation in observations])
    features = torch.from_numpy(feature_array).to(device=device, dtype=torch.float32)
    reach = _pad_token_sequences(
        [observation.reach for observation in observations],
        device=device,
    )
    avoid = _pad_token_sequences(
        [observation.avoid for observation in observations],
        device=device,
    )
    epsilon_mask = torch.tensor(
        [observation.epsilon_enabled for observation in observations],
        dtype=torch.bool,
        device=device,
    )
    return torch_ac.DictList(
        {
            "features": features,
            "seq": (reach, avoid),
            "epsilon_mask": epsilon_mask,
        }
    )


class ReplayBuffer:
    """Small deterministic replay buffer with object observations."""

    def __init__(self, capacity: int, *, seed: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._data: list[Transition] = []
        self._next_index = 0

    def __len__(self) -> int:
        return len(self._data)

    def append(self, transition: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            self._data[self._next_index] = transition
        self._next_index = (self._next_index + 1) % self.capacity

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._data):
            raise ValueError("cannot sample more transitions than the buffer contains")
        # Sampling directly from the ring avoids allocating/copying a list of
        # up to 200,000 references on every gradient update.
        return self._rng.sample(self._data, batch_size)


def epsilon_at_step(config: DQNConfig, step: int) -> float:
    """Linear epsilon schedule used by the frozen DQN learner."""

    if step <= 0:
        return config.initial_epsilon
    fraction = min(float(step) / float(config.epsilon_decay_steps), 1.0)
    return config.initial_epsilon + fraction * (config.final_epsilon - config.initial_epsilon)


def valid_action_indices(action_dim: int, *, epsilon_enabled: bool) -> tuple[int, ...]:
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    base = tuple(range(action_dim))
    return (*base, action_dim) if epsilon_enabled else base


def action_index_to_env_action(action_index: int, action_dim: int) -> int:
    """Map the Q-head's extra epsilon category to DeepLTL's sentinel action."""

    if action_index == action_dim:
        return EPSILON_ACTION
    if 0 <= action_index < action_dim:
        return action_index
    raise ValueError(f"invalid action index {action_index} for action_dim={action_dim}")


def env_action_to_action_index(action: int, action_dim: int) -> int:
    if action == EPSILON_ACTION:
        return action_dim
    if 0 <= action < action_dim:
        return action
    raise ValueError(f"invalid environment action {action} for action_dim={action_dim}")


def nonterminal_indices(transitions: Sequence[Transition]) -> tuple[int, ...]:
    """Return batch positions whose next value should be bootstrapped."""

    return tuple(index for index, transition in enumerate(transitions) if not transition.done)


def choose_epsilon_greedy_index(
    q_values: Sequence[float],
    *,
    action_dim: int,
    epsilon_enabled: bool,
    epsilon: float,
    rng: random.Random,
) -> int:
    """Choose a valid action index from masked Q-values.

    ``q_values`` has length ``action_dim + 1``.  The final element is the
    optional LDBA epsilon transition category.
    """

    if len(q_values) != action_dim + 1:
        raise ValueError("q_values must contain action_dim normal values plus epsilon")
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    valid = valid_action_indices(action_dim, epsilon_enabled=epsilon_enabled)
    if rng.random() < epsilon:
        return rng.choice(valid)
    best = max(valid, key=lambda index: (float(q_values[index]), -index))
    return int(best)


def resolve_device(requested: str):
    torch = _require_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def build_q_network(env: Any, config: DQNConfig):
    """Build the LetterEnv Q-network using DeepLTL's published encoders."""

    torch = _require_torch()
    import torch.nn as nn
    import config as deepltl_config
    from model.ltl.ltl_net import LTLNet
    from preprocessing.vocab import VOCAB
    from utils import torch_utils

    model_config = deepltl_config.model_configs["LetterEnv-v0"]
    obs_shape = env.observation_space["features"].shape
    action_dim = int(env.action_space.n)
    if model_config.env_net is None:
        env_net = None
        env_embedding_dim = int(obs_shape[0])
    else:
        env_net = model_config.env_net.build(obs_shape)
        env_embedding_dim = env_net.embedding_size
    if len(VOCAB) <= 3:
        raise ValueError("DeepLTL vocabulary is not initialized")
    embedding = nn.Embedding(len(VOCAB), model_config.ltl_embedding_dim, padding_idx=VOCAB["PAD"])
    ltl_net = LTLNet(embedding, model_config.set_net, model_config.num_rnn_layers)
    head = torch_utils.make_mlp_layers(
        [env_embedding_dim + ltl_net.embedding_dim, 64, 64, action_dim + 1],
        activation=nn.ReLU,
        final_layer_activation=False,
    )
    return LetterDQNNetwork(env_net, ltl_net, head, action_dim)


class LetterDQNNetwork:
    """Torch module wrapper, constructed lazily by ``build_q_network``."""

    def __new__(cls, env_net: Any, ltl_net: Any, head: Any, action_dim: int):
        torch = _require_torch()
        import torch.nn as nn

        class _Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.env_net = env_net
                self.ltl_net = ltl_net
                self.head = head
                self.action_dim = action_dim

            def forward(self, obs: Any):
                env_embedding = (
                    self.env_net(obs.features)
                    if self.env_net is not None
                    else obs.features
                )
                ltl_embedding = self.ltl_net(obs.seq)
                q_values = self.head(torch.cat([env_embedding, ltl_embedding], dim=1))
                masked = q_values.clone()
                masked[~obs.epsilon_mask, self.action_dim] = float("-inf")
                return masked

        return _Network()


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LetterEnv DQN runtime requires PyTorch. Use the DeepLTL-compatible "
            "environment, for example a Python 3.10 environment set up with pip install -e .[benchmarks]."
        ) from exc
    return torch
