"""EA4LLM optimizer: gradient-free evolution strategies for LLM optimization.

EA4LLM (Liu et al. 2025, arXiv:2510.10603) replaces backpropagation with a
population-based evolution-strategies (ES) estimator.  Each step perturbs the
parameters with ``population`` scaled Gaussian noises, scores every perturbed
point on the current minibatch's next-token log-likelihood fitness (the
negated training loss), shapes the rewards into weights by z-score
normalization or centered ranks, and moves the parameters along the
noise-weighted mean direction.  No gradient is consumed and no per-parameter
state is kept, so the optimizer's memory footprint is O(1) in the parameter
count.

The step is *closure-based*, following the ``torch.optim`` LBFGS convention:
the trainer supplies a callable that re-evaluates the current batch at the
live parameter values and returns the loss as a float.  The trainer routes
any optimizer exposing ``requires_closure = True`` through that path
automatically (shared with EvE), which is why EA4LLM requires
``gradient_accumulation_steps: 1`` and is not available for SBERT.

Two-pass seeded replay keeps the estimator's memory at O(1): one integer seed
is drawn per ES member, the noise is regenerated deterministically from that
seed in both the evaluation pass and the direction-accumulation pass, so the
full noise set never has to be stored.

Reference:
    Liu, W., Song, S., Hao, H., & Zhou, A. (2025). EA4LLM: A Gradient-Free
    Approach to Large Language Model Optimization via Evolutionary
    Algorithms. arXiv:2510.10603.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


class EA4LLM(Optimizer):
    r"""Gradient-free evolution strategies for large language model training.

    Implements Algorithm 1 of "EA4LLM: A Gradient-Free Approach to Large
    Language Model Optimization via Evolutionary Algorithms"
    (arXiv:2510.10603).  The fitness of a parameter vector is the negated
    current-batch loss (the paper's token-level mean log-probability of the
    true next token), so maximising the fitness descends the training loss.

    Each iteration draws ``population`` independent Gaussian noises
    :math:`\epsilon_j \sim \mathcal{N}(0, I)`, evaluates
    :math:`r_j = -f(\theta + \sigma \epsilon_j)` on the trainer's current
    minibatch, converts the rewards into weights
    :math:`w_j = (r_j - \bar r)/\operatorname{std}(r)` (default) or centered
    ranks :math:`w_j = \operatorname{rank}(r_j) - (N-1)/2`, and applies
    :math:`\theta \leftarrow \theta + \alpha \hat g` with
    :math:`\hat g = \frac{1}{N}\sum_j w_j \epsilon_j` (or the NES-style
    :math:`\frac{1}{N\sigma}\sum_j w_j \epsilon_j`).  Antithetic sampling
    evaluates the :math:`\pm \sigma \epsilon` pair of each seed.

    Args:
        params: Iterable of parameters to optimize or dicts defining groups.
        lr: Step size :math:`\alpha` (default: 5e-4, the paper's value).  The
            factory routes the shared per-group ``lr_*`` suffixes here, so the
            effective step size is the parameter group's ``lr``.
        population: ES population size :math:`N` (default: 30).
        sigma: Gaussian noise scale :math:`\sigma` (default: 1e-3).
        antithetic: Evaluate the :math:`\pm \sigma \epsilon` pairs of
            ``population // 2`` seeds instead of one sign per seed
            (default: ``False``; requires an even population).
        rank_shaping: Use centered-rank fitness shaping instead of z-score
            normalization (default: ``False``).
        nes_scaling: Divide the aggregated direction by :math:`\sigma`
            (NES-style scaling, default: ``False``).
        weight_decay: Decoupled multiplicative decay applied before the ES
            move.  The paper defines no weight decay; this is a
            repo-consistency extension configured by the shared ``wd_*``
            suffixes (default: 0.0).
        seed: Integer seed for the persistent generator (reproducible noise).
            When ``None``, a seed is drawn once from the global torch RNG at
            construction time.

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter state.  Only the first parameter carries
            scalar bookkeeping (``step``); no moment buffers are allocated,
            which is the O(1)-state property reported in the paper.
        requires_closure (bool): Class-level marker read by the trainer to
            route this optimizer through the closure-based step path.
    """

    requires_closure = True

    def __init__(
        self,
        params,
        lr: float = 5e-4,
        population: int = 30,
        sigma: float = 1e-3,
        antithetic: bool = False,
        rank_shaping: bool = False,
        nes_scaling: bool = False,
        weight_decay: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the EA4LLM optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            lr: Step size (default: 5e-4).
            population: ES population size (default: 30).
            sigma: Gaussian noise scale (default: 1e-3).
            antithetic: Evaluate antithetic ``+/-`` pairs (default: ``False``).
            rank_shaping: Use centered-rank shaping instead of z-score
                (default: ``False``).
            nes_scaling: Use NES-style ``1/sigma`` scaling (default:
                ``False``).
            weight_decay: Decoupled decay coefficient (default: 0.0).
            seed: RNG seed for the persistent generator, or ``None`` to draw
                one from the global torch RNG.

        Raises:
            ValueError: If a hyper-parameter is out of its valid range, or if
                ``antithetic`` is requested with an odd population.
        """
        if not 0.0 <= float(lr):
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= float(weight_decay):
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if int(population) < 2:
            raise ValueError(f"Invalid population: {population} (must be >= 2)")
        if not 0.0 < float(sigma):
            raise ValueError(f"Invalid sigma: {sigma} (must be > 0)")
        if bool(antithetic) and int(population) % 2 != 0:
            raise ValueError(
                f"Invalid population: {population} (antithetic sampling requires an even population)"
            )

        defaults = dict(
            lr=float(lr),
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=float(weight_decay),
            population=int(population),
            sigma=float(sigma),
            antithetic=bool(antithetic),
            rank_shaping=bool(rank_shaping),
            nes_scaling=bool(nes_scaling),
        )
        super().__init__(params, defaults)

        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.seed = int(seed)
        self._cpu_gen = torch.Generator(device="cpu")
        self._cpu_gen.manual_seed(self.seed)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate(closure) -> float:
        """Call the closure once at the live parameter values.

        Args:
            closure: Callable that zeroes gradients, runs the task forward and
                backward pass on the current batch and returns the loss
                (float or 0-dim tensor).  The backward pass is unnecessary for
                EA4LLM but harmless: the optimizer never reads the gradients.

        Returns:
            The loss as a float.
        """
        with torch.enable_grad():
            value = closure()
        if torch.is_tensor(value):
            value = value.detach().item()
        return float(value)

    @staticmethod
    def _randn_from_seed(seed: int, like: Tensor) -> Tensor:
        """Draw standard-normal noise reproducibly from an integer seed.

        A fresh generator is seeded per member and per pass so the same noise
        is recreated without storing the full tensor set (two-pass replay).

        Args:
            seed: Integer seed identifying the ES member.
            like: Tensor whose shape, device and dtype are matched.

        Returns:
            A noise tensor shaped like ``like`` in its dtype.
        """
        try:
            generator = torch.Generator(device=like.device)
            generator.manual_seed(int(seed))
            noise = torch.randn(
                like.shape, generator=generator, device=like.device, dtype=torch.float32
            )
        except (RuntimeError, TypeError):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
            noise = torch.randn(
                like.shape, generator=generator, device="cpu", dtype=torch.float32
            ).to(device=like.device)
        return noise.to(dtype=like.dtype)

    @staticmethod
    def _weights(rewards: List[float], rank_shaping: bool) -> List[float]:
        """Convert rewards into normalized ES weights.

        Implements the paper's two normalization choices: z-score
        normalization (default) or rank-based shaping with centered ranks
        ``rank - (N - 1) / 2`` (Salimans et al. 2017).  Non-finite rewards
        receive a zero weight so a single failed evaluation cannot corrupt
        the direction.

        Args:
            rewards: Fitness values, one per ES member.
            rank_shaping: Use centered ranks instead of the z-score.

        Returns:
            A list of float weights aligned with ``rewards``.
        """
        count = len(rewards)
        finite_indices = [index for index, value in enumerate(rewards) if math.isfinite(value)]
        if not finite_indices:
            return [0.0] * count

        if rank_shaping:
            order = sorted(finite_indices, key=rewards.__getitem__)
            center = (len(order) - 1) / 2.0
            shaped = [0.0] * count
            for rank, index in enumerate(order):
                shaped[index] = float(rank) - center
            return shaped

        mean = sum(rewards[index] for index in finite_indices) / len(finite_indices)
        variance = (
            sum((rewards[index] - mean) ** 2 for index in finite_indices)
            / len(finite_indices)
        )
        std = math.sqrt(variance)
        if std <= 0.0:
            return [0.0] * count
        return [
            (value - mean) / std if math.isfinite(value) else 0.0
            for value in rewards
        ]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Performs one evolution-strategies iteration.

        Executes Algorithm 1 of the paper on the trainer's current minibatch:
        draw one seed per member (pass 1), perturb the live parameters by
        ``+/- sigma * eps``, read the negated loss as the fitness, restore the
        parameters, shape the rewards into weights, then replay the same noise
        from the seeds (pass 2) to accumulate the weighted direction and apply
        ``theta <- theta + alpha * g_hat`` per parameter group.

        Args:
            closure: Callable that zeroes gradients, runs the task forward and
                backward pass on the current batch at the live parameter
                values and returns the loss as a float.  Required — EA4LLM
                evaluates ``population`` candidate points per iteration.

        Returns:
            The mean fitness (negated loss) over the population as a float, or
            ``None`` when the optimizer has no parameters.

        Raises:
            RuntimeError: If ``closure`` is ``None``.
        """
        if closure is None:
            raise RuntimeError(
                "EA4LLM requires a closure-based step: pass a callable that zeroes "
                "gradients, runs the forward pass on the current batch and returns "
                "the loss as a float. The Frankenstein trainer supplies it "
                "automatically for optimizer_class=ea4llm."
            )

        params: List[Tensor] = [p for group in self.param_groups for p in group["params"]]
        if not params:
            return None

        first_state = self.state[params[0]]
        first_state["step"] = int(first_state.get("step", 0)) + 1

        group0 = self.param_groups[0]
        population = int(group0["population"])
        sigma = float(group0["sigma"])
        antithetic = bool(group0["antithetic"])
        rank_shaping = bool(group0["rank_shaping"])
        nes_scaling = bool(group0["nes_scaling"])

        # Decoupled multiplicative decay (repo extension; the paper defines
        # no weight decay, and the shared wd_* default is 0.0).
        for group in self.param_groups:
            decay = float(group["weight_decay"])
            if decay > 0.0:
                step_size = float(group["lr"])
                for p in group["params"]:
                    p.mul_(1.0 - step_size * decay)

        # One reproducible integer seed per ES member (two-pass replay keeps
        # the optimizer state O(1): the noise tensors are never stored).
        members = population // 2 if antithetic else population
        seeds = [
            int(torch.randint(0, 2**31 - 1, (1,), generator=self._cpu_gen).item())
            for _ in range(members)
        ]
        signs: Tuple[float, ...] = (1.0, -1.0) if antithetic else (1.0,)

        # Pass 1: perturb, evaluate the negated loss (fitness), restore.
        rewards: List[float] = []
        for seed in seeds:
            for sign in signs:
                noise: Dict[int, Tensor] = {}
                for p in params:
                    eps = self._randn_from_seed(seed, p)
                    noise[id(p)] = eps
                    p.add_(eps, alpha=sign * sigma)
                try:
                    reward = -self._evaluate(closure)
                finally:
                    for p in params:
                        p.add_(noise[id(p)], alpha=-sign * sigma)
                rewards.append(reward)

        weights = self._weights(rewards, rank_shaping)
        if not any(weight != 0.0 for weight in weights):
            return float(sum(rewards) / len(rewards)) if rewards else None

        # Pass 2: replay the same noise from the seeds and accumulate the
        # weighted direction into one transient buffer per parameter.
        directions: Dict[int, Tensor] = {id(p): torch.zeros_like(p) for p in params}
        member = 0
        for seed in seeds:
            for sign in signs:
                weight = weights[member]
                member += 1
                if weight == 0.0:
                    continue
                for p in params:
                    directions[id(p)].add_(
                        self._randn_from_seed(seed, p), alpha=weight * sign
                    )

        # Apply theta <- theta + lr * g_hat (g_hat is 1/N sum w_j eps_j, or the
        # NES 1/(N sigma) variant).  Per-group lr is the paper's alpha.
        scale = 1.0 / (population * (sigma if nes_scaling else 1.0))
        for group in self.param_groups:
            step_size = float(group["lr"])
            for p in group["params"]:
                p.add_(directions[id(p)], alpha=step_size * scale)

        return float(sum(rewards) / len(rewards)) if rewards else None
