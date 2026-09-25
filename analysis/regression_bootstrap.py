"""Vectorized Newton refits of fixed whole-puzzle bootstrap draws.

Batching shares array operations only: each fit has its own coefficients,
gradient stopping decision and Armijo step length. Every estimate is retained.
"""
import math
import numpy as np
from . import DEFAULT_L2_PENALTY
from scipy.special import expit, logsumexp
from .regression_optimizer import (GRADIENT_TOLERANCE, MAX_ITERATIONS,
                                   penalty_metadata, validate_l2_penalty)


def training_features(x, mask, multiplicities, source):
    tensor = np.broadcast_to(x, (len(multiplicities), *x.shape))
    if not source:
        return tensor
    maximum = np.max(np.where(mask[:, :, None], x, -np.inf), axis=1)
    minimum = np.min(np.where(mask[:, :, None], x, np.inf), axis=1)
    constant = maximum == minimum
    puzzle_mean = np.sum(np.where(mask[:, :, None], x, 0), axis=1) / mask.sum(axis=1)[:, None]
    weights = multiplicities[:, :, None] * ~constant[None]
    denominator = weights.sum(axis=1)
    mean = np.divide(np.sum(weights * puzzle_mean[None], axis=1), denominator,
                     out=np.full(denominator.shape, .5), where=denominator > 0)
    tensor = np.where(constant[None, :, None, :], mean[:, None, None, :], tensor)
    return np.concatenate((np.ones((*tensor.shape[:-1], 1)), tensor), axis=-1)


def batch_fit(x, mask, q, multiplicities, human=None, *, l2_penalty=DEFAULT_L2_PENALTY):
    """Return every coefficient vector and a diagnostic for each training set."""
    l2_penalty = validate_l2_penalty(l2_penalty)
    source = human is not None
    features = training_features(x, mask, multiplicities, source)
    weights = np.divide(multiplicities, multiplicities.sum(axis=1)[:, None],
                        out=np.zeros_like(multiplicities, dtype=float),
                        where=multiplicities.sum(axis=1)[:, None] > 0)
    beta = np.zeros((len(multiplicities), features.shape[-1]))
    penalized = np.ones(beta.shape[-1])
    if source:
        penalized[0] = 0.0
    status = np.full(len(beta), 'iteration_budget', dtype=object)
    iterations = np.zeros(len(beta), dtype=int)

    def evaluate(values, indices, hessian=False):
        xx = features[indices]; ww = weights[indices]
        scores = np.einsum('btkd,bd->btk', xx, values)
        if source:
            mass = (human + q) / 2
            probabilities = expit(scores)
            loss = np.einsum('bt,btk->b', ww,
                mass[None] * np.logaddexp(0, scores) - q[None] / 2 * scores)
            gradient = np.einsum('bt,btk,btkd->bd', ww, mass[None] * probabilities - q[None] / 2, xx)
            if hessian:
                h = np.einsum('bt,btk,btkd,btke->bde', ww,
                    mass[None] * probabilities * (1-probabilities), xx, xx)
        else:
            logz = logsumexp(np.where(mask[None], scores, -np.inf), axis=2)
            probabilities = np.where(mask[None], np.exp(scores-logz[:, :, None]), 0.)
            loss = np.einsum('bt,bt->b', ww, logz - np.sum(q[None] * scores, axis=2))
            gradient = np.einsum('bt,btk,btkd->bd', ww, probabilities-q[None], xx)
            if hessian:
                mean = np.einsum('btk,btkd->btd', probabilities, xx)
                h = (np.einsum('bt,btk,btkd,btke->bde', ww, probabilities, xx, xx)
                     - np.einsum('bt,btd,bte->bde', ww, mean, mean))
        if l2_penalty:
            loss += .5 * l2_penalty * np.sum(penalized * values**2, axis=1)
            gradient += l2_penalty * penalized * values
            if hessian:
                h += np.diag(l2_penalty * penalized)[None]
        return (loss, gradient, h) if hessian else (loss, gradient)

    active = np.arange(len(beta))
    for iteration in range(MAX_ITERATIONS):
        loss, gradient, hessian = evaluate(beta[active], active, True)
        done = np.max(np.abs(gradient), axis=1) <= GRADIENT_TOLERANCE
        iterations[active] = iteration + 1
        status[active[done]] = 'gradient_tolerance'
        active = active[~done]
        if not len(active):
            break
        loss, gradient, hessian = loss[~done], gradient[~done], hessian[~done]
        # Singular designs take the minimum-norm Newton direction.
        try:
            step = -np.linalg.solve(hessian, gradient[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            step = -np.einsum('bij,bj->bi', np.linalg.pinv(hessian, hermitian=True), gradient)
        slope = np.sum(gradient * step, axis=1)
        fallback = ~np.isfinite(step).all(axis=1) | (slope >= 0)
        step[fallback] = -gradient[fallback]
        slope = np.sum(gradient * step, axis=1)
        scale = np.ones(len(active)); pending = np.arange(len(active))
        for _ in range(60):
            proposed = beta[active[pending]] + scale[pending, None] * step[pending]
            new_loss, _ = evaluate(proposed, active[pending])
            accept = (np.isfinite(proposed).all(axis=1) & np.isfinite(new_loss)
                      & (new_loss <= loss[pending] + 1e-4*scale[pending]*slope[pending] + 1e-15))
            beta[active[pending[accept]]] = proposed[accept]
            pending = pending[~accept]
            if not len(pending):
                break
            scale[pending] *= .5
        if len(pending):
            status[active[pending]] = 'line_search_limit'
            active = np.delete(active, pending)
            if not len(active):
                break
    loss, gradient = evaluate(beta, np.arange(len(beta)))
    diagnostic = [{'method': 'damped_newton', 'penalty': penalty_metadata(l2_penalty),
        'gradient_inf': float(np.max(abs(g))),
        'gradient_tolerance_met': bool(np.max(abs(g)) <= GRADIENT_TOLERANCE),
        'iterations': int(i), 'status': str(s), 'estimate_retained': True,
        'objective': float(f)} for f, g, i, s in zip(loss, gradient, iterations, status)]
    return beta, diagnostic


def refit_all(x, mask, q, multiplicities, human=None, *, l2_penalty=DEFAULT_L2_PENALTY):
    values=[]; diagnostics=[]
    for start in range(0, len(multiplicities), 128):
        beta, info = batch_fit(x, mask, q, multiplicities[start:start+128], human, l2_penalty=l2_penalty)
        values.append(beta); diagnostics.extend(info)
    return np.vstack(values), diagnostics


def bootstrap_validation(x, mask, q, rng, repeats, human=None, *, l2_penalty=DEFAULT_L2_PENALTY):
    """Same fixed draws for coefficients and full identity-safe validation."""
    n = len(x)
    samples = rng.integers(0, n, size=(repeats, n))
    multiplicity = np.sum(samples[:, :, None] == np.arange(n)[None, None, :], axis=1).astype(float)
    coefficients, diagnostics = refit_all(x, mask, q, multiplicity, human, l2_penalty=l2_penalty)
    source = human is not None
    fields = ['log_loss', 'log_loss_gain', 'normalized_accuracy']
    fields += (['balanced_accuracy', 'balanced_accuracy_relative_to_uniform'] if source else
               ['baseline_log_loss', 'top_choice_accuracy', 'baseline_top_choice_accuracy', 'top_choice_accuracy_relative_to_uniform'])
    summary = {key: np.zeros(repeats) for key in fields}
    for held in range(n):
        selected = np.flatnonzero(multiplicity[:, held] > 0)
        if not len(selected):
            continue
        train = multiplicity[selected].copy(); train[:, held] = 0
        beta, info = refit_all(x, mask, q, train, human, l2_penalty=l2_penalty)
        diagnostics.extend(info)
        xx = training_features(x, mask, train, source)[:, held]
        scores = np.einsum('bkd,bd->bk', xx, beta)
        valid = mask[held]
        if source:
            probs = np.clip(expit(scores), 1e-12, 1-1e-12)
            loss = -.5*np.sum(human[held]*np.log1p(-probs)+q[held]*np.log(probs), axis=1)
            accuracy = .5*np.sum(human[held]*(probs < .5)+q[held]*(probs >= .5), axis=1)
            baseline = .5
            gap = .25 * np.abs(q[held]-human[held]).sum()
            metrics = {'log_loss': loss, 'log_loss_gain': math.log(2)-loss,
                       'balanced_accuracy': accuracy, 'balanced_accuracy_relative_to_uniform': accuracy/.5}
        else:
            scores = np.where(valid[None], scores, -np.inf)
            probs = np.exp(scores-logsumexp(scores, axis=1)[:, None])
            top = np.isclose(probs, probs.max(axis=1)[:, None], rtol=1e-12, atol=1e-12) & valid[None]
            accuracy = np.sum(q[held]*top, axis=1)/top.sum(axis=1)
            baseline = 1/valid.sum(); gap = q[held].max()-baseline
            loss = -np.sum(q[held]*np.log(np.clip(probs, 1e-12, 1)), axis=1)
            metrics = {'log_loss': loss, 'baseline_log_loss': math.log(valid.sum()),
                       'log_loss_gain': math.log(valid.sum())-loss, 'top_choice_accuracy': accuracy,
                       'baseline_top_choice_accuracy': baseline, 'top_choice_accuracy_relative_to_uniform': accuracy/baseline}
        metrics['normalized_accuracy'] = (accuracy-baseline)/gap if gap > 1e-12 else np.zeros(len(selected))
        for key, value in metrics.items():
            summary[key][selected] += multiplicity[selected, held]/n * value
    validation = [{key: float(value[i]) for key, value in summary.items()} for i in range(repeats)]
    return coefficients[:, 1:] if source else coefficients, validation, diagnostics
