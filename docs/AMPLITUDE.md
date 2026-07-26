# LINEX Amplitude Estimation

## Scope

This note isolates the estimator mathematics from the surrounding RIR pipeline. Decay times are treated as known; the unknowns are non-negative energy amplitudes. For the numerical continuation solver, see [Robust α-continuation](LINEX_CONVERGENCE_FIX.md).

## Signal and energy model

Let $h_i=h(t_i)$ be a sampled room impulse response. A stochastic multi-slope model is

\[
h_i=\sum_{j=1}^{d} A_j\exp(-\gamma_j t_i)z_{ij}, \qquad z_{ij}\sim\mathcal N(0,1),
\]

with independent carriers $z_{ij}$. A reverberation time $T_j$ specifies a 60 dB amplitude drop:

\[
\gamma_j=\frac{3\ln 10}{T_j}.
\]

Since independent variances add, the expected instantaneous energy is

\[
m_i=\mathbb E[h_i^2] =\sum_{j=1}^{d}w_j\exp(-\delta_jt_i), \qquad w_j=A_j^2,\quad \delta_j=2\gamma_j=\frac{\ln(10^6)}{T_j}.
\]

Define $X_{ij}=\exp(-\delta_jt_i)$, so $\mathbf{m}=\mathbf{Xw}$, with $w_j\geq0$. Setting $T_j=\infty$ gives $\delta_j=0$ and a constant column representing stationary background noise. The reported level is

\[
10\log_{10}w_j=20\log_{10}A_j.
\]

## LINEX objective

For observed energy $y_i=h_i^2$, define the log-energy residual

\[
e_i=\log y_i-\log m_i=\log\frac{y_i}{(\mathbf{Xw})_i}.
\]

The LINEX loss with asymmetry parameter $\alpha>0$ is

\[
\ell_\alpha(e)=\exp(\alpha e)-\alpha e-1,
\]

and the estimate is

\[
\widehat w =\arg\min_{0\leq w\leq u} J_\alpha(w), \qquad J_\alpha(w)=\sum_i\ell_\alpha(e_i).
\]

This asymmetry makes the fit follow the stochastic energy envelope rather than the dense cluster of near-zero squared samples.

If $h_i\sim\mathcal N(0,m_i)$, minimizing its negative log-likelihood is equivalent, up to constants and a factor, to minimizing $J_\alpha(w)$ when $\alpha=1$. **Thus the published setting has a variance maximum-likelihood interpretation under the independent Gaussian-carrier model.** Octave filtering introduces temporal correlation, so this becomes a quasi-likelihood interpretation for filtered RIRs.

## Assumptions and porting cautions

- Decay times $T_j$ are fixed upstream. Errors or duplicates in $T_j$ transfer directly to amplitude bias and ill-conditioning.
- Input to the core solver is instantaneous energy $h^2$. The convenience API squares a raw signal. Schroeder-integrated energy follows a different observation model.
- Each time sample has equal weight. Correlation, changing effective degrees of freedom, truncation, and late noise-floor samples are currently unmodeled.
- Octave-filtered amplitudes are in-band energy levels. Their absolute levels can differ from broadband ground truth while slope-to-slope comparisons remain meaningful.

## Code correspondence and provenance

The equations map to `multi_slope_linex/kernel.py`, `linex.py`, and `estimator.py`; synthetic validation is in `synth.py` and `tests/test_recovery.py`. The numerical procedure is documented separately in [Robust α-continuation](LINEX_CONVERGENCE_FIX.md). The LINEX amplitude method and continuation solver are Jeremy Bai's contribution. The surrounding common-slope model and pipeline derive partly from Georg Götz's work. Consult `THIRD_PARTY_NOTICES.md` before publication.
