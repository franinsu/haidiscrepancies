"""Deterministic logits with optional L2 slopes; retain every finite incumbent."""
from dataclasses import dataclass
import numpy as np
from . import DEFAULT_L2_PENALTY
from scipy.special import expit, logsumexp

GRADIENT_TOLERANCE = 1e-8
MAX_ITERATIONS = 10000  # Safety budget, never a sample-selection criterion.


def validate_l2_penalty(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError('l2_penalty must be finite and nonnegative')
    return value


def penalty_metadata(l2_penalty=DEFAULT_L2_PENALTY):
    strength = validate_l2_penalty(l2_penalty)
    return ({'kind': 'l2', 'strength': strength, 'intercept_penalized': False}
            if strength else None)


@dataclass
class LogitObjective:
    task: str
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    mask: np.ndarray | None = None
    l2_penalty: float = DEFAULT_L2_PENALTY

    def __post_init__(self):
        self.l2_penalty = validate_l2_penalty(self.l2_penalty)
        self.weights = np.asarray(self.weights, dtype=float)
        total = self.weights.sum()
        if total > 0:
            self.weights = self.weights / total

    def evaluate(self, beta, hessian=False):
        scores = self.x @ beta
        if self.task == 'choice':
            logz = logsumexp(np.where(self.mask, scores, -np.inf), axis=1)
            probs = np.where(self.mask, np.exp(scores - logz[:, None]), 0.0)
            loss = self.weights @ (logz - np.sum(self.y * scores, axis=1))
            grad = np.einsum('t,tj,tjd->d', self.weights, probs - self.y, self.x)
            if hessian:
                means = np.einsum('tj,tjd->td', probs, self.x)
                h = (np.einsum('t,tj,tjd,tje->de', self.weights, probs, self.x, self.x)
                     - np.einsum('t,td,te->de', self.weights, means, means))
        else:
            probs = expit(scores)
            loss = self.weights @ (np.logaddexp(0, scores) - self.y * scores)
            grad = self.x.T @ (self.weights * (probs - self.y))
            if hessian:
                h = self.x.T @ ((self.weights * probs * (1 - probs))[:, None] * self.x)
        if self.l2_penalty:
            # Source designs begin with an intercept; choice designs do not.
            penalized = np.ones(len(beta))
            if self.task == 'source':
                penalized[0] = 0.0
            loss += .5 * self.l2_penalty * (penalized * beta) @ beta
            grad += self.l2_penalty * penalized * beta
            if hessian:
                h += np.diag(self.l2_penalty * penalized)
        if hessian:
            return float(loss), grad, h
        return float(loss), grad


def damped_newton(objective, *, tolerance=GRADIENT_TOLERANCE, max_iterations=MAX_ITERATIONS):
    beta = np.zeros(objective.x.shape[-1])
    status = 'iteration_budget'
    iteration = 0
    for iteration in range(max_iterations):
        loss, grad, hessian = objective.evaluate(beta, True)
        if np.max(np.abs(grad)) <= tolerance:
            status = 'gradient_tolerance'
            break
        try:
            step = -np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(hessian, hermitian=True) @ grad
        if not np.isfinite(step).all() or grad @ step >= 0:
            step = -grad
        scale = 1.0
        for _ in range(60):
            proposal = beta + scale * step
            if np.isfinite(proposal).all():
                new_loss, _ = objective.evaluate(proposal)
                if np.isfinite(new_loss) and new_loss <= loss + 1e-4 * scale * (grad @ step) + 1e-15:
                    beta = proposal
                    break
            scale *= 0.5
        else:
            status = 'line_search_limit'
            break
    loss, grad, hessian = objective.evaluate(beta, True)
    diagnostic = {
        'method': 'damped_newton', 'penalty': penalty_metadata(objective.l2_penalty),
        'objective': loss, 'gradient_inf': float(np.max(np.abs(grad))),
        'gradient_tolerance_met': bool(np.max(np.abs(grad)) <= tolerance),
        'iterations': iteration + 1, 'status': status,
        'estimate_retained': True,
    }
    return beta, hessian, diagnostic


def method_metadata(l2_penalty=DEFAULT_L2_PENALTY):
    penalty = penalty_metadata(l2_penalty)
    return {
        'method': 'damped_newton', 'penalty': penalty,
        'objective': ('weighted mean negative log likelihood + (strength / 2) * sum(feature slopes squared)'
                      if penalty else 'weighted mean negative log likelihood'),
        'gradient_tolerance': GRADIENT_TOLERANCE,
        'line_search': 'Armijo backtracking, c=1e-4, step halving',
        'safety_max_iterations': MAX_ITERATIONS,
        'retention': 'Every finite incumbent retained; no fit or puzzle deletion based on optimizer diagnostics.',
    }


def finalize_regression_metadata(data, l2_penalty=DEFAULT_L2_PENALTY):
    repeats = data.get('bootstrap_repeats', data.get('repeats', 2000))
    data['optimizer'] = method_metadata(l2_penalty)
    data['validation_scheme'] = {
        'point_estimates': 'Leave one entire puzzle out; average held-out metrics equally.',
        'bootstrap_intervals': f'95% percentile intervals from {repeats:,} fixed whole-puzzle draws, with complete validation refit in each draw.',
        'holdout_rule': 'Remove all copies of a held-out puzzle from training.',
        'aggregation': 'Weight validation puzzles by bootstrap multiplicity.',
        'paired_with_coefficients': 'Identical fixed draws for coefficients and validation.',
        'retention': 'All draws retained, regardless of optimization diagnostics; no replacement draws.',
    }
    data['rng_seed_rule'] = 'Independent generator with the recorded common seed for every fit; exactly the prescribed draws, without replacements.'
    data['constant_feature_policy'] = 'Training mean over nonconstant puzzles with bootstrap multiplicities; re-estimated after holding out all puzzle copies; 0.5 if no nonconstant training puzzle remains.'
    for key in ['selected_solution_accuracy_estimand', 'source_accuracy_estimand']:
        if key in data:
            data[key]['interval'] = data['validation_scheme']['bootstrap_intervals']
    return data
