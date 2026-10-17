"""GTBM (Game-Theoretic Block Model): a constrained SBM with a game-theoretic mixing matrix.

A 2-block (K-block) Stochastic Block Model whose mixing matrix is not free but
derived from evolutionary game-theoretic partner-choice equations. Inference
recovers both the community assignments and interpretable payoff parameters
from a single observed graph — no neural network, no training data.

Estimators (EM and Gibbs):
    fit            variational EM (constrained, game-theoretic)
    sweep_beta     Brent's-method search for the ELBO-maximising beta (EM)
    fit_gibbs      Gibbs sampling (constrained, full Bayesian)
    fit_free       variational EM (free / unconstrained baseline)
    fit_free_gibbs Gibbs sampling (free baseline)

Model-comparison diagnostics (as reported in the paper):
    ELBO                     returned by fit / fit_free
    compute_model_comparison MDL-BIC, MDL-Laplace, Vuong test
    compute_delta_elpd       k-fold cross-validated ΔELPD
    waic_from_loglik         WAIC from Gibbs posterior draws
    compare_loo              PSIS-LOO-CV (needs optional `arviz`)
    r_hat / effective_sample_size  MCMC convergence diagnostics
"""

# Core model (partner-choice equations and M -> B conversion)
from constrained_sbm.model import (
    compute_M,
    M_to_B,
    compute_M_conditional,
    getGameType,
    getGraphType,
)

# Inference engines
from constrained_sbm.variational_em import fit
from constrained_sbm.beta_sweep import sweep_beta
from constrained_sbm.baseline_sbm import fit_free
from constrained_sbm.posterior import (
    fit_gibbs,
    r_hat,
    effective_sample_size,
)
from constrained_sbm.posterior_free import fit_free_gibbs

# Interpretation helpers
from constrained_sbm.get_game_labels import game_label
from constrained_sbm.canonicalize import canonicalize_result, canonicalize_orientation

# Model-comparison diagnostics
from constrained_sbm.evaluation import compare_models
from constrained_sbm.mdl import (
    compute_model_comparison,
    compute_delta_elpd,
    kass_raftery_category,
)
from constrained_sbm.waic import (
    pointwise_loglik_constrained,
    pointwise_loglik_free,
    waic_from_loglik,
    compare_waic,
    loo_from_loglik,
    compare_loo,
)

__all__ = [
    # model
    "compute_M",
    "M_to_B",
    "compute_M_conditional",
    "getGameType",
    "getGraphType",
    # inference
    "fit",
    "sweep_beta",
    "fit_free",
    "fit_gibbs",
    "fit_free_gibbs",
    "r_hat",
    "effective_sample_size",
    # interpretation
    "game_label",
    "canonicalize_result",
    "canonicalize_orientation",
    # diagnostics
    "compare_models",
    "compute_model_comparison",
    "compute_delta_elpd",
    "kass_raftery_category",
    "pointwise_loglik_constrained",
    "pointwise_loglik_free",
    "waic_from_loglik",
    "compare_waic",
    "loo_from_loglik",
    "compare_loo",
]
