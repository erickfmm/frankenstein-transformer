"""EvE optimizer: steady-state differential evolution with a targeted Adam fallback.

EvE (Raj & Deb 2026, arXiv:2609.35614) maintains a steady-state population of
four candidate solutions.  Each iteration challenges one round-robin slot with
a ``rand-to-pbest/1`` differential-evolution child evaluated on the current
minibatch (common random numbers); only when the child fails to improve the
slot's incumbent does EvE run a short burst of stock Adam — persistent
per-slot moments, classic L2 weight decay folded into the gradient, results
clamped back into the search box — keeping the refined point only if it beats
the child.  Greedy selection makes the best-so-far value monotone
non-increasing on deterministic objectives, while per-iteration cost stays
within a constant factor of a single Adam step regardless of dimension.

Following the paper's neural-network recipe, the population is anchored at the
model's own initial weights with the other three members placed as Gaussian
jitters around the anchor (the LoRA precedent of Appendix E), the per-tensor
search box is centered on the anchor with the Kaiming half-width
``sqrt(6 / fan_in)`` (Appendix C), and polynomial mutation is off by default
(Section 3.4: it never wins its accept/reject gate at network scale).

Reference:
    Raj, S., & Deb, K. (2026). EVE: An Alternate Optimizer to Adam.
    arXiv:2609.35614.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Optimizer

_N_POP = 4
"""Population size, fixed by the paper (Section 3.1); deliberately not a knob."""


class Eve(Optimizer):
    """EvE optimizer: population-of-four DE with a targeted Adam fallback.

    Implements Algorithm 1 of "EVE: An Alternate Optimizer to Adam"
    (arXiv:2609.35614).  Unlike every other optimizer in this package, EvE is
    *closure-based*: :meth:`step` requires a callable that zeroes gradients,
    runs forward+backward on the current batch at the live parameter values and
    returns the loss as a float, because one iteration scores several candidate
    points (Algorithm 1 lines 6, 9 and 21).  The trainer supplies the closure
    automatically for ``optimizer_class: eve``; between steps the live
    parameters hold the best-so-far slot, so validation and checkpointing
    always see Algorithm 1 line 26's return point.

    Per-parameter state holds three stacked slot tensors (``slots``, ``m``,
    ``v``, leading dimension 4) plus the per-tensor box bounds, i.e. 12x the
    parameter count in optimizer state (13x with the live parameters), the
    fixed dimension-independent multiplier reported in the paper (Section 4).
    Slot losses, per-slot Adam step counters and the iteration counter live in
    the first parameter's state so they serialize with ``state_dict``.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.  Per-group ``lr``/``weight_decay``/``betas``/
            ``eps`` configure the Adam fallback for that group's parameters;
            ``weight_decay`` is a classic L2 penalty folded into the fallback
            gradient (the paper's "PyTorch's stock Adam"), not AdamW's
            decoupled rule.
        de_weight: DE base weight ``F`` (default: 0.5).  The per-iteration
            jitter applies only to the difference term; the pbest pull uses
            the unjittered ``F``.
        de_jitter: Multiplicative jitter ``gamma`` on ``F_t = F (1 + gamma
            (u - 0.5))`` with ``u ~ U(0, 1)`` (default: 1e-4).
        init_sigma: Proximity-initialization jitter ``sigma``: the three
            non-anchor population members are drawn as Gaussian jitters around
            the anchor (default: 0.1).
        adam_steps: Fallback burst length ``n_adam`` (default: 3, the paper's
            neural-network setting; the synthetic benchmarks used 7).
        use_pm: Whether to apply polynomial mutation before the fallback
            decision (default: ``False``; the paper turns PM off for neural
            network training).
        pm_eta: Polynomial-mutation distribution index ``eta`` (default: 20).
        seed: Integer seed for the two persistent generators (paper Section 4,
            "Reproducibility").  When ``None``, a seed is drawn once from the
            global torch RNG — which ``training.seed`` controls — at
            construction time.

    Attributes:
        defaults (dict): Default hyper-parameter values for each parameter
            group.
        param_groups (list): Parameter groups tracked by the optimizer.
        state (dict): Per-parameter optimizer state (stacked slot tensors and
            box bounds; the first parameter additionally carries the slot
            losses, per-slot Adam counters and the iteration counter).
        requires_closure (bool): Class-level marker read by the trainer to
            route this optimizer through the closure-based step path.
    """

    requires_closure = True

    def __init__(
        self,
        params,
        de_weight: float = 0.5,
        de_jitter: float = 1e-4,
        init_sigma: float = 0.1,
        adam_steps: int = 3,
        use_pm: bool = False,
        pm_eta: float = 20.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the EvE optimizer.

        Args:
            params: Iterable of parameters to optimize or dicts defining
                parameter groups.
            de_weight: DE base weight ``F`` (default: 0.5).
            de_jitter: Multiplicative jitter on ``F_t`` (default: 1e-4).
            init_sigma: Proximity-init jitter around the anchor
                (default: 0.1).
            adam_steps: Adam-fallback burst length (default: 3).
            use_pm: Enable polynomial mutation (default: ``False``).
            pm_eta: Polynomial-mutation distribution index (default: 20).
            seed: RNG seed for the persistent generators, or ``None`` to draw
                one from the global torch RNG.

        Raises:
            ValueError: If any hyper-parameter is out of its valid range.
        """
        if float(de_weight) < 0.0:
            raise ValueError(f"Invalid de_weight: {de_weight}")
        if float(de_jitter) < 0.0:
            raise ValueError(f"Invalid de_jitter: {de_jitter}")
        if float(init_sigma) < 0.0:
            raise ValueError(f"Invalid init_sigma: {init_sigma}")
        if int(adam_steps) < 1:
            raise ValueError(f"Invalid adam_steps: {adam_steps}")
        if float(pm_eta) <= 0.0:
            raise ValueError(f"Invalid pm_eta: {pm_eta}")

        defaults = dict(
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            de_weight=float(de_weight),
            de_jitter=float(de_jitter),
            init_sigma=float(init_sigma),
            adam_steps=int(adam_steps),
            use_pm=bool(use_pm),
            pm_eta=float(pm_eta),
        )
        super().__init__(params, defaults)

        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.seed = int(seed)
        self._cpu_gen = torch.Generator(device="cpu")
        self._cpu_gen.manual_seed(self.seed)
        self._dev_gen: Optional[torch.Generator] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bounds(param: torch.Tensor):
        """Compute the per-tensor search box centered on the anchor weights.

        Follows the paper's Appendix C convention: the box has the Kaiming
        half-width ``b = sqrt(6 / fan_in)`` per tensor, the default the paper
        uses for any network it is not told an explicit domain for.  For
        tensors with fewer than two dimensions the fan-in is approximated by
        the element count (the paper's "a bias inherits its weight tensor's
        fan-in" is not knowable from inside an optimizer).

        Args:
            param: Parameter tensor at its anchor (initial) values.

        Returns:
            Tuple ``(x_l, x_u)`` of lower/upper bound tensors.
        """
        if param.ndim >= 2:
            try:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(param)
            except (ValueError, TypeError):
                fan_in = param.numel()
        else:
            fan_in = param.numel()
        half_width = math.sqrt(6.0 / max(float(fan_in), 1.0))
        center = param.detach()
        return center - half_width, center + half_width

    def _randn_like(self, param: torch.Tensor) -> torch.Tensor:
        """Draw standard-normal noise matching ``param``'s shape and dtype.

        Uses the persistent device generator (paper Section 4) when the
        parameter lives on that device, falling back to the CPU generator with
        a device transfer for exotic mixed-device models.

        Args:
            param: Tensor whose shape, device and dtype are matched.

        Returns:
            Noise tensor shaped like ``param`` in its dtype.
        """
        if self._dev_gen is not None and self._dev_gen.device == param.device:
            noise = torch.randn(
                param.shape, generator=self._dev_gen, device=param.device, dtype=torch.float32
            )
        else:
            noise = torch.randn(
                param.shape, generator=self._cpu_gen, device="cpu", dtype=torch.float32
            ).to(param.device)
        return noise.to(param.dtype)

    def _init_slots(self, params: List[torch.Tensor]) -> None:
        """Build the population, box and moment state for every parameter.

        Slot 0 is the anchor (the model's current weights, the Appendix E
        precedent for pre-initialized models); slots 1-3 are Gaussian jitters
        around it, clamped into the box (Eq. 1).  Adam moment buffers are
        allocated zero-initialized so fallback bursts share one continuous
        trajectory per slot (Section 3.5).

        Args:
            params: Flat list of optimizer-owned parameters.
        """
        first = params[0]
        if self._dev_gen is None:
            self._dev_gen = torch.Generator(device=first.device)
            self._dev_gen.manual_seed(self.seed)

        sigma = float(self.param_groups[0]["init_sigma"])
        for p in params:
            state = self.state[p]
            if "slots" in state:
                continue
            x_l, x_u = self._bounds(p)
            span = x_u - x_l
            anchor = p.detach().clone()
            slots = anchor.unsqueeze(0).repeat((_N_POP,) + (1,) * p.ndim)
            for k in range(1, _N_POP):
                jitter = sigma * span * self._randn_like(p)
                slots[k] = torch.clamp(anchor + jitter, x_l, x_u)
            state["x_l"] = x_l
            state["x_u"] = x_u
            state["slots"] = slots
            state["m"] = torch.zeros_like(slots)
            state["v"] = torch.zeros_like(slots)

        state0 = self.state[first]
        if "slot_losses" not in state0:
            # Python list (not a tensor) so Optimizer.load_state_dict's dtype
            # casting never touches the recorded losses.
            state0["slot_losses"] = [float("inf")] * _N_POP
        if "slot_t" not in state0:
            state0["slot_t"] = [0] * _N_POP
        state0.setdefault("iter", 0)

    @staticmethod
    def _evaluate(closure) -> float:
        """Call the closure once at the live parameter values.

        Args:
            closure: Callable that zeroes gradients, runs forward+backward on
                the current batch and returns the loss (float or 0-dim tensor).

        Returns:
            The loss as a float.
        """
        with torch.enable_grad():
            value = closure()
        if torch.is_tensor(value):
            value = value.detach().item()
        return float(value)

    def _load_slot(self, params: List[torch.Tensor], slot: int) -> None:
        """Copy one population slot into the live parameters.

        Args:
            params: Flat list of optimizer-owned parameters.
            slot: Slot index in ``[0, _N_POP)``.
        """
        for p in params:
            p.data.copy_(self.state[p]["slots"][slot])

    @staticmethod
    def _load_tensors(params: List[torch.Tensor], tensors: Dict[int, torch.Tensor]) -> None:
        """Copy arbitrary candidate coordinates into the live parameters.

        Args:
            params: Flat list of optimizer-owned parameters.
            tensors: Mapping ``id(param) -> candidate tensor``.
        """
        for p in params:
            p.data.copy_(tensors[id(p)])

    def _polynomial_mutation(
        self,
        params: List[torch.Tensor],
        child: Dict[int, torch.Tensor],
        eta: float,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """Apply the paper's Eq. (3) polynomial mutation to one coordinate.

        Mutates exactly one gene (``k = max(1, round(n * p)) = 1`` at the
        classical rate ``p = 1/n``), chosen uniformly over all parameter
        coordinates.  The perturbation is bounded and self-scaling to the
        distance to each boundary, so no clipping is required.  The caller
        evaluates the mutant and keeps it only if strictly better (PM's own
        accept/reject gate, Section 3.4).

        Args:
            params: Flat list of optimizer-owned parameters.
            child: Mapping ``id(param) -> child coordinates`` (real space).
            eta: Distribution index; larger values concentrate the
                perturbation near the parent.

        Returns:
            New mapping with the mutated parameter replaced, or ``None`` when
            there are no coordinates to mutate.
        """
        total = sum(p.numel() for p in params)
        if total <= 0:
            return None
        pick = float(torch.rand(1, generator=self._cpu_gen).item()) * total
        target = None
        target_idx = 0
        acc = 0
        for p in params:
            n = p.numel()
            if pick < acc + n:
                target = p
                target_idx = int(pick - acc)
                break
            acc += n
        if target is None:
            return None

        state = self.state[target]
        x_l, x_u = state["x_l"], state["x_u"]
        span = x_u - x_l
        z_child = (child[id(target)] - x_l) / span
        z_flat = z_child.reshape(-1).clamp_(0.0, 1.0).clone()
        z_d = float(z_flat[target_idx])
        # Unit-cube bounds: delta1 = (z_d - 0)/(1 - 0), delta2 = (1 - z_d).
        delta1 = z_d
        delta2 = 1.0 - z_d
        r = float(torch.rand(1, generator=self._cpu_gen).item())
        exponent = 1.0 / (eta + 1.0)
        if r <= 0.5:
            inner = 2.0 * r + (1.0 - 2.0 * r) * (1.0 - delta1) ** exponent
            delta_q = inner**exponent - 1.0
        else:
            inner = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (1.0 - delta2) ** exponent
            delta_q = 1.0 - inner**exponent
        z_flat[target_idx] = min(max(z_d + delta_q, 0.0), 1.0)  # (z_u - z_l) = 1
        x_new = x_l + z_flat.reshape(z_child.shape) * span

        mutated: Dict[int, torch.Tensor] = {}
        for p in params:
            mutated[id(p)] = x_new if p is target else child[id(p)].clone()
        return mutated

    def _adam_burst(self, slot: int, n_adam: int, closure) -> None:
        """Run the persistent per-slot Adam fallback burst (Eqs. 4-6).

        Performs ``n_adam`` stock-Adam steps on the live parameters (which
        hold the failed child in real coordinates), using the challenged
        slot's persistent ``(m, v, t)`` state so bursts form one continuous
        Adam trajectory across the run.  Classic L2 weight decay is folded
        into the gradient before the moment update; every iterate is clamped
        back into the per-tensor box.

        Args:
            slot: Challenged slot index owning the moment state.
            n_adam: Number of Adam steps in the burst.
            closure: Loss closure re-evaluated before every step; each call
                leaves fresh gradients on the parameters.
        """
        first_param = self._first_param()
        for _ in range(n_adam):
            self._evaluate(closure)
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                eps = group["eps"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    m = state["m"][slot]
                    v = state["v"][slot]
                    grad = p.grad
                    if wd != 0:
                        grad = grad.add(p.detach(), alpha=wd)  # L2 into the gradient
                    first_state = self.state[first_param]
                    t = int(first_state["slot_t"][slot]) + 1
                    first_state["slot_t"][slot] = t

                    m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    m_hat = m / (1.0 - beta1**t)
                    v_hat = v / (1.0 - beta2**t)
                    denom = v_hat.sqrt_().add_(eps)
                    p.data.addcdiv_(m_hat, denom, value=-lr)
                    p.data.clamp_(state["x_l"], state["x_u"])

    def _first_param(self) -> torch.Tensor:
        """Return the first optimizer-owned parameter.

        Returns:
            The first parameter of the first non-empty parameter group.
        """
        for group in self.param_groups:
            for p in group["params"]:
                return p
        raise RuntimeError("Eve has no parameters to optimize")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Performs one steady-state Evolutionary-Explorer iteration.

        Executes Algorithm 1 of the paper at the current minibatch: refresh
        the challenged slot (line 6), propose a DE child (lines 7-9),
        optionally mutate it under PM's accept/reject gate (lines 10-13),
        rescue it with the persistent Adam burst when it fails (lines 14-22),
        and accept the candidate only if it strictly beats the incumbent
        (line 24).  The live parameters end the step holding the best-so-far
        slot (line 26 applied continuously).

        Args:
            closure: Callable that zeroes gradients, runs forward+backward on
                the current batch at the live parameter values and returns the
                loss as a float.  Required — EvE scores several candidate
                points per iteration.

        Returns:
            The best-so-far slot loss as a float, or ``None`` when the
            optimizer has no parameters.

        Raises:
            RuntimeError: If ``closure`` is ``None``.
        """
        if closure is None:
            raise RuntimeError(
                "Eve requires a closure-based step: pass a callable that zeroes "
                "gradients, runs forward+backward on the current batch and returns "
                "the loss as a float. The Frankenstein trainer supplies it "
                "automatically for optimizer_class=eve."
            )

        params: List[torch.Tensor] = [
            p for group in self.param_groups for p in group["params"]
        ]
        if not params:
            return None

        first_state = self.state[params[0]]

        # Algorithm 1, lines 1-3: population + initial evaluations.
        if "slots" not in first_state:
            self._init_slots(params)
            for k in range(_N_POP):
                self._load_slot(params, k)
                value = self._evaluate(closure)
                first_state["slot_losses"][k] = value if math.isfinite(value) else float("inf")

        first_state["iter"] = int(first_state.get("iter", 0)) + 1
        t_iter = int(first_state["iter"])
        i = (t_iter - 1) % _N_POP  # round-robin challenged slot (line 5)

        # Line 6: refresh the challenged slot's record on this batch.
        self._load_slot(params, i)
        refresh = self._evaluate(closure)
        if math.isfinite(refresh):
            first_state["slot_losses"][i] = refresh

        group0 = self.param_groups[0]
        f_base = float(group0["de_weight"])
        gamma = float(group0["de_jitter"])
        use_pm = bool(group0["use_pm"])
        pm_eta = float(group0["pm_eta"])
        n_adam = int(group0["adam_steps"])

        # Line 7: distinct donors and the jittered difference weight.
        others = [k for k in range(_N_POP) if k != i]
        perm = torch.randperm(_N_POP - 1, generator=self._cpu_gen).tolist()
        a, b, c = others[perm[0]], others[perm[1]], others[perm[2]]
        u = float(torch.rand(1, generator=self._cpu_gen).item())
        f_t = f_base * (1.0 + gamma * (u - 0.5))

        slot_losses = first_state["slot_losses"]
        best = min(range(_N_POP), key=slot_losses.__getitem__)

        # Line 8: rand-to-pbest/1 child, built in the unit cube.
        child: Dict[int, torch.Tensor] = {}
        for p in params:
            state = self.state[p]
            x_l, x_u = state["x_l"], state["x_u"]
            span = x_u - x_l
            z = (state["slots"] - x_l) / span
            z_child = z[a] + f_t * (z[b] - z[c]) + f_base * (z[best] - z[a])
            z_child = z_child.clamp_(0.0, 1.0)
            child[id(p)] = x_l + z_child * span

        # Line 9: evaluate the child on the current batch.
        self._load_tensors(params, child)
        f_child = self._evaluate(closure)

        # Lines 10-13: PM's own accept/reject gate (optional).
        if use_pm and math.isfinite(f_child):
            mutated = self._polynomial_mutation(params, child, pm_eta)
            if mutated is not None:
                self._load_tensors(params, mutated)
                f_mut = self._evaluate(closure)
                if math.isfinite(f_mut) and f_mut < f_child:
                    child = mutated
                    f_child = f_mut
                    self._load_tensors(params, child)

        # Lines 14-23: accept the child, or rescue it with the Adam burst.
        f_i = float(slot_losses[i])
        f_cand = f_child
        if not (math.isfinite(f_child) and f_child < f_i):
            self._adam_burst(i, n_adam, closure)
            f_adam = self._evaluate(closure)
            if math.isfinite(f_adam) and math.isfinite(f_child) and f_adam < f_child:
                f_cand = f_adam
            else:
                self._load_tensors(params, child)

        # Line 24: greedy, accept-only-if-strictly-better selection.
        if math.isfinite(f_cand) and f_cand < f_i:
            for p in params:
                self.state[p]["slots"][i].copy_(p.detach())
            slot_losses[i] = f_cand

        # Line 26, applied continuously: the model holds the best-so-far slot.
        best = min(range(_N_POP), key=slot_losses.__getitem__)
        self._load_slot(params, best)
        return float(slot_losses[best])