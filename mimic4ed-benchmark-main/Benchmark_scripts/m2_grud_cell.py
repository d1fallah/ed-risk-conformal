"""
The GRU-D recurrent cell -- single shared definition.

Extracted from m2_hybrid_common.py so that BOTH M2 pipelines use one
implementation rather than two hand-copied ones:
  * m2_hybrid_common.py  -- the v3.8 controlled five-model comparison
  * m2_ablations.py      -- the Track A mandatory core (GRU-D + MAPIE vs
                            NEWS2, and the roadmap's required ablations)

m2_hybrid_common.py imports several helpers FROM m2_ablations.py, so
m2_ablations.py cannot import from m2_hybrid_common.py at module level
without a cycle. This module depends on neither and breaks the knot.

See GRUDCell's own docstring for the mechanism, the bug it replaces, and
why its initialisation is load-bearing.
"""
import torch
import torch.nn as nn


class GRUDCell(nn.Module):
    """The GRU-D recurrent cell, implemented per Che, Purushotham, Cho,
    Sontag & Liu, "Recurrent Neural Networks for Multivariate Time Series
    with Missing Values," Scientific Reports 8:6085, 2018 (Eqs. 10-13).

    Both GRUDWithEncoder (Stage 1, trained) and TemporalEncoder (Stages
    2/3, frozen) hold an instance of THIS class rather than each carrying
    their own copy of the recurrence, so the trained and the frozen paths
    are guaranteed to compute the same function -- previously they were
    two hand-copied blocks that could drift.

    ------------------------------------------------------------------
    Why this was rewritten (M2_Hybrid_Deviation_Log.md Deviation 7)
    ------------------------------------------------------------------
    The previous implementation was:

        gamma = exp(-relu(d @ W_gamma + b_gamma))
        xc    = m * x + (1 - m) * (gamma * x)

    Because prepare_temporal_tensors() encodes missing entries as x = 0,
    the second term evaluated to (1-m) * (gamma * 0) = 0 for every missing
    entry, and the first term is zeroed wherever m = 0. So `gamma` was
    multiplied by zero everywhere it appeared: it could not affect the
    output, W_gamma/b_gamma received exactly zero gradient, and (being
    zero-initialised) they stayed exactly zero for the whole of training.
    Verified directly on all 5 saved seed checkpoints of the previous run:
    W_gamma absmax = 0.000e+00, b_gamma absmax = 0.000e+00, and
    |out(D) - out(1000*D)| = 0.0 in eval mode. Two consequences:
      * the decay mechanism -- GRU-D's entire reason for being chosen over
        a plain GRU -- had never been active in any M2 run; and
      * Delta-t never reached the model at all, since D enters ONLY
        through gamma. The model was a masked GRU with zero-imputation.

    The correct formulation, now implemented:

        gamma_x = exp(-relu(w_x * delta + b_x))          (diagonal, per input)
        gamma_h = exp(-relu(W_h delta + b_h))            (full, per hidden unit)
        x_hat   = m*x + (1-m)*(gamma_x * x_last + (1-gamma_x) * x_tilde)
        h       = gamma_h * h                            (hidden-state decay)

    where x_last is the last actually-observed value of each variable
    (carried forward) and x_tilde is the training-set empirical mean. In
    the scaled space produced by prepare_temporal_tensors(), x_tilde = 0
    exactly, so the interpolation term is just (1-gamma_x)*0 -- the mean is
    represented, not dropped. x_last is initialised to 0 (= x_tilde) so a
    variable never yet observed decays toward the mean, per the paper.

    Initialisation note, and it matters: the decay parameters are NOT
    zero-initialised. PyTorch defines relu'(0) = 0, so zero-init makes
    z = w*delta + b identically 0, relu passes zero gradient, and the decay
    parameters would freeze at zero forever -- reproducing the exact bug
    this rewrite exists to fix, just by a different route. w_x is drawn
    N(0, 0.1) and gamma_h uses nn.Linear's default init; biases start at 0
    so that at triage (delta = 0) gamma = exp(-relu(0)) = 1, i.e. no decay
    when no time has elapsed, which is the behaviour the paper specifies.
    """

    def __init__(self, inp=7, hid=32):
        super().__init__()
        self.inp, self.hid = inp, hid
        # gamma_x: Che et al. constrain W_gamma_x to be DIAGONAL, so it is
        # parameterised here as a length-`inp` vector, not an (inp, inp)
        # matrix (the previous code used a full matrix, a second departure
        # from the paper).
        self.gamma_x_w = nn.Parameter(torch.randn(inp) * 0.1)
        self.gamma_x_b = nn.Parameter(torch.zeros(inp))
        # gamma_h: full matrix, mapping the per-variable delta vector to a
        # per-hidden-unit decay.
        self.gamma_h = nn.Linear(inp, hid)
        nn.init.zeros_(self.gamma_h.bias)
        self.W_z = nn.Linear(inp + hid + inp, hid)
        self.W_r = nn.Linear(inp + hid + inp, hid)
        self.W_h = nn.Linear(inp + hid + inp, hid)

    def step(self, x, m, d, h, x_last):
        """One recurrence step (the body of the forward() loop below,
        extracted so it can also be driven externally by
        run_grud_masked() for variable-length/padded sequences -- added
        2026-09-19 for the irregular-series cohort rebuild
        (M2_Outcome_Definition_Gate.md v2 Sec 4). Pure refactor: forward()
        below calls this in a loop and is verified to produce identical
        output to the pre-refactor single-block version, so every
        existing checkpoint (GRUD_CELL_PARAM_KEYS are parameter names,
        untouched by this refactor) loads and behaves exactly as before.
        Returns (new_h, new_x_last)."""
        gamma_x = torch.exp(-torch.relu(self.gamma_x_w * d + self.gamma_x_b))
        gamma_h = torch.exp(-torch.relu(self.gamma_h(d)))
        # x_tilde is 0 in scaled space, so the (1-gamma_x)*x_tilde term
        # is written out but contributes nothing numerically; kept
        # explicit so the code matches the paper if the scaling
        # convention ever changes.
        x_hat = m * x + (1 - m) * (gamma_x * x_last)
        h = gamma_h * h
        gi = torch.cat([x_hat, h, m], 1)
        z = torch.sigmoid(self.W_z(gi))
        r = torch.sigmoid(self.W_r(gi))
        ht = torch.tanh(self.W_h(torch.cat([x_hat, r * h, m], 1)))
        new_h = (1 - z) * h + z * ht
        new_x_last = m * x + (1 - m) * x_last
        return new_h, new_x_last

    def forward(self, X, M, D):
        B, T, _ = X.size()
        h = torch.zeros(B, self.hid, device=X.device)
        # x_last == x_tilde == 0 in the scaled space (see class docstring).
        x_last = torch.zeros(B, self.inp, device=X.device)
        for t in range(T):
            h, x_last = self.step(X[:, t, :], M[:, t, :], D[:, t, :], h, x_last)
        return h


def run_grud_masked(cell, X, M, D, pad_mask):
    """Drives `cell` over a right-padded, variable-length batch (X, M, D
    each (B, T, inp), `pad_mask` (B, T) 1.0 at real timesteps, 0.0 at
    padding). Added for the irregular-series cohort rebuild
    (M2_Outcome_Definition_Gate.md v2 Sec 4), which replaces the fixed
    T=2 (triage + last) representation with a genuine variable-length ED-
    vitalsign sequence, T_max=10 (+triage), truncated/padded to a common
    T=11 for batching.

    A padding timestep has obs_mask all-zero there BY CONSTRUCTION (see
    m2_build_irregular_cohort.py::build_irregular_sequences), so feeding
    it through cell.step() unmodified would still be "valid" in the sense
    that GRU-D always tolerates M=0 (that's the whole point of the decay
    mechanism) -- but it would let the recurrence keep evolving `h` past
    each sequence's real end, and worse, return h from a FIXED final
    index (T-1) that is only meaningful for the longest sequences in the
    batch. Every shorter sequence's true final state would be silently
    overwritten by additional padded steps.

    Fix: at each timestep t, only apply cell.step()'s update to samples
    still inside their real length (pad_mask[:,t]==1); samples that have
    already ended keep last step's (h, x_last) frozen via a masked blend.
    This is the standard masked/packed-sequence technique for variable-
    length RNNs, applied here rather than sorting+packing since GRU-D's
    per-variable decay state (x_last) also needs the same freeze, which
    torch's built-in pack_padded_sequence does not know how to carry.

    Returns h: (B, hid), each row the cell's hidden state at ITS OWN last
    real (non-padded) timestep -- directly comparable across a batch of
    differing true lengths.
    """
    B, T, _ = X.size()
    h = torch.zeros(B, cell.hid, device=X.device)
    x_last = torch.zeros(B, cell.inp, device=X.device)
    for t in range(T):
        step_h, step_x_last = cell.step(X[:, t, :], M[:, t, :], D[:, t, :], h, x_last)
        alive = pad_mask[:, t:t + 1]  # (B,1), broadcasts over hid/inp
        h = alive * step_h + (1 - alive) * h
        x_last = alive * step_x_last + (1 - alive) * x_last
    return h


GRUD_CELL_PARAM_KEYS = (
    "cell.gamma_x_w", "cell.gamma_x_b", "cell.gamma_h.weight", "cell.gamma_h.bias",
    "cell.W_z.weight", "cell.W_z.bias", "cell.W_r.weight", "cell.W_r.bias",
    "cell.W_h.weight", "cell.W_h.bias",
)
