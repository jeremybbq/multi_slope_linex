# Robust α-Continuation

## Scope

This note documents the numerical procedure used by `constrained_linex_decay_analysis`. It solves the LINEX amplitude objective from [Basic estimator mathematics](LINEX_AMPLITUDE_MATH.md) with fixed decay times, and deal with convergence problems.

## Why direct optimization is fragile

A direct solve at the target $\alpha=1$ can encounter very large positive log-energy residuals, poorly scaled kernel columns, zero crossings in squared RIR samples, and a poor flat initialization. Together these effects can create overflow, unstable gradients, or convergence to a poor local solution.

For small $\alpha$,

\[
\tilde\ell_\alpha(e)\approx\frac{\alpha^2}{2}e^2+O(\alpha^3).
\]

The scaled objective

\[
\widetilde J_\alpha(w) =\frac{2}{\alpha^2}\sum_i\tilde\ell_\alpha(e_i)
\]

therefore approaches squared log-residual fitting as $\alpha\to0$. Scaling from $\alpha=0$ offers a good initialisation with simple least square fit and keeps the objective and termination tolerances comparable across the schedule.

## Procedure

For each energy curve, the solver:

1. Uses non-negative least squares on $\mathbf{Xw}\approx \mathbf{y}$ for the initial point.
2. Solves LINEX loss minimization with L-BFGS-B on a geometric schedule from $\alpha=0.05$ to $1$, warm-starting each from the preceding result. The normalized bounds are $0\leq w_j\leq10$.
3. Evaluates $g(u)=\exp(u)-u-1$ with `expm1`. For $u>30$, it uses a $C^1$ linear continuation, keeping the loss and gradient finite.
4. Retries with a longer schedule and several positive starts when the final state is non-finite.

The last subproblem is the requested $\alpha=1$ objective. Earlier subproblems only provide a stable path to its solution.

## Regression check

`tests/test_recovery.py::test_continuation_beats_direct_solve` compares the default schedule with a one-step $\alpha=1$ solve using the same NNLS warm start. On its seeded weak-slope case, continuation reaches at least 98 of 100 successful fits with maximum amplitude RMSE below 1.5 dB; the direct solve is required to perform substantially worse.

## Implementation correspondence

The schedule, normalization, bounds, and fallback are in `multi_slope_linex/estimator.py`; the overflow-safe loss is in `multi_slope_linex/linex.py`. The published method is described in Bai and Schlecht, [*Estimation of Multi-Slope Amplitudes in Late Reverberation*](https://www.dafx.de/paper-archive/2025/DAFx25_paper_28.pdf), DAFx25, 2025, pp. 194–201.
