"""OpenFAWE safeguarded Newton solution of an unchanged circulation closure.

This selects a numerical root finder, not a different aerodynamic model.
Finite-difference Newton follows the approach used in KiteFAST VSM.f90;
domain-aware backtracking is an OpenFAWE safeguard. No SciPy dependency.
"""

import json
import numpy as np


class NonlinearSolveError(RuntimeError):
    """No admissible, decreasing Newton step or no converged circulation root."""

    def __init__(self, message, *, iterations=None, diagnostics=None):
        super().__init__(message)
        self.iterations = iterations
        self.diagnostics = diagnostics


def solve_newton(target, initial, settings, *, jacobian=None, diagnostics=None):
    """Return only a root satisfying the caller's original residual tolerances.

    ``target(gamma)`` must reject invalid physical states with ValueError.
    Trial states outside that domain are rejected, never clipped or accepted.
    The caller validates shape/finite input before entering this function.
    A Jacobian callback returns d(target-Gamma)/dGamma at accepted iterates.
    Callback errors are input errors and are never retried as FD or Picard.
    Optional diagnostics receive actual iterates without changing acceptance.
    """
    if jacobian is not None and not callable(jacobian):
        raise ValueError('Newton Jacobian callback must be callable')
    if settings.jacobian != 'finite_difference' and jacobian is None:
        raise ValueError('selected Newton Jacobian requires a callback')
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise ValueError('Newton diagnostics must be a dictionary')
    if diagnostics is not None:
        diagnostics.clear()
    gamma = np.array(initial, dtype=float, copy=True)

    def failure(message, iteration):
        return NonlinearSolveError(message, iterations=iteration, diagnostics=diagnostics)

    def observe(updates):
        if diagnostics is not None:
            diagnostics.update(updates=updates, circulation=gamma.copy(), residual=error.copy(),
                               merit=float(error@error))

    def residual(value):
        result = np.asarray(target(value)) - value
        if result.shape != gamma.shape or not np.isfinite(result).all():
            raise ValueError('invalid nonlinear circulation residual')
        return result

    error = residual(gamma)
    observe(0)
    for iteration in range(1, settings.max_iterations + 1):
        maximum = float(np.max(np.abs(error)))
        limit = settings.absolute_tolerance + settings.relative_tolerance * float(np.mean(np.abs(gamma)))
        if maximum <= limit:
            return gamma, iteration
        if jacobian is None:
            matrix = np.empty((len(gamma), len(gamma)))
            for j in range(len(gamma)):
                step = np.cbrt(np.finfo(float).eps) * max(1., abs(gamma[j]))
                for shrink in range(12):
                    delta = np.zeros_like(gamma)
                    delta[j] = step * 2.**-shrink
                    try:
                        matrix[:, j] = (residual(gamma + delta) - error) / delta[j]
                        break
                    except ValueError:
                        try:
                            matrix[:, j] = (error - residual(gamma - delta)) / delta[j]
                            break
                        except ValueError:
                            continue
                else:
                    raise failure('Newton derivative has no admissible perturbation', iteration)
        else:
            value = np.asarray(jacobian(gamma))
            if value.dtype.kind not in 'fiu' or value.shape != (len(gamma), len(gamma)) or not np.isfinite(value).all():
                raise ValueError('invalid Newton Jacobian callback result')
            matrix = np.asarray(value, dtype=float)
        linear_solve = 'solve'
        try:
            direction = np.linalg.solve(matrix, -error)
        except np.linalg.LinAlgError:
            linear_solve = 'lstsq'
            try:
                direction = np.linalg.lstsq(matrix, -error, rcond=None)[0]
            except np.linalg.LinAlgError as linear_failure:
                raise NonlinearSolveError('Newton linear algebra failed', iterations=iteration,
                                          diagnostics=diagnostics) from linear_failure
        merit = float(error @ error)
        domain_rejected = no_decrease = 0
        if diagnostics is not None:
            diagnostics.update(jacobian=matrix.copy(), direction=direction.copy(),
                               linearization_circulation=gamma.copy(), linear_solve=linear_solve)
        for reduction in range(32):
            fraction = 2.**-reduction
            trial = gamma + fraction * direction
            try:
                trial_error = residual(trial)
            except ValueError:
                domain_rejected += 1
                continue
            if float(trial_error @ trial_error) <= (1. - 1e-4 * fraction) * merit:
                gamma, error = trial, trial_error
                observe(iteration)
                break
            no_decrease += 1
        else:
            if diagnostics is not None:
                diagnostics.update(domain_rejected=domain_rejected, no_decrease=no_decrease)
            raise failure(f'Newton line search failed at residual {maximum:g} m^2/s', iteration)
        if diagnostics is not None:
            diagnostics.setdefault('history', []).append(dict(update=iteration,
                accepted_fraction=fraction, domain_rejected=domain_rejected, no_decrease=no_decrease))
    # The final allowed update is a candidate too; do not report its old residual.
    maximum = float(np.max(np.abs(error)))
    limit = settings.absolute_tolerance + settings.relative_tolerance * float(np.mean(np.abs(gamma)))
    if maximum <= limit:
        return gamma, iteration
    raise failure(f'Newton failed after {iteration} iterations: residual {maximum:g} > {limit:g} m^2/s', iteration)


class HybridSolveError(NonlinearSolveError):
    """All declared attempts failed; the caller must not advance its state."""

    def __init__(self, attempts, *, diagnostics=None):
        self.attempts_json = json.dumps(attempts, sort_keys=True, allow_nan=False)
        super().__init__('Newton/Picard attempts exhausted: '+self.attempts_json, diagnostics=diagnostics)


class HybridLoadReevaluationError(NonlinearSolveError):
    """Preserve a successful root's trace when the load reevaluation rejects it."""

    def __init__(self, trace_json, residual, limit, *, diagnostics=None):
        self.solver_trace_json = trace_json
        self.residual, self.limit = residual, limit
        super().__init__(f'hybrid load-evaluation residual {residual:g} > {limit:g} m^2/s; '
                         f'accepted solver trace: {trace_json}', diagnostics=diagnostics)


def solve_newton_picard(target, initial, settings, *, jacobian=None, diagnostics=None):
    """Opt-in Newton-first residual-root selection, with deterministic retries.

    Each Picard attempt restarts at the unchanged original seed. This is an
    algebraic selection policy, not a physical stability or uniqueness test.
    Budgets count updates; the last allowed update is checked for acceptance.
    """
    seed = np.array(initial, dtype=float, copy=True)
    if seed.ndim != 1 or not seed.size or not np.isfinite(seed).all():
        raise ValueError('invalid hybrid circulation seed')

    def residual(value):
        error = np.asarray(target(value))-value
        if error.shape != seed.shape or not np.isfinite(error).all():
            raise ValueError('invalid nonlinear circulation residual')
        return error

    def metrics(value, error):
        return (float(np.max(abs(error))), float(settings.absolute_tolerance+
                settings.relative_tolerance*np.mean(abs(value))))

    # Invalid input is not a convergence failure and must not trigger retries.
    residual(seed)
    attempts = []

    def accepted(value, iterations, attempt):
        maximum, limit = metrics(value, residual(value))
        if maximum > limit:
            raise ValueError('hybrid result failed original residual criterion')
        attempt.update(status='accepted', iterations=iterations,
                       residual_m2_s=maximum, residual_limit_m2_s=limit)
        attempts.append(attempt)
        trace = dict(schema='newton_picard_v1', policy='newton_first_residual_root_v1',
            selected_solver=attempt['solver'], relaxation=attempt.get('relaxation', 0.),
            fallback_used=attempt['solver']=='picard', attempts=attempts,
            delta_seed_linf_m2_s=float(np.max(abs(value-seed))),
            stability='not_enforced; offline active-segment diagnostic required')
        if settings.jacobian != 'finite_difference':
            trace['jacobian'] = settings.jacobian
            trace['updates'] = (diagnostics['updates'] if attempt['solver']=='newton'
                                else iterations-1)
        return value, iterations, json.dumps(trace, sort_keys=True, allow_nan=False)

    try:
        if settings.jacobian != 'finite_difference' and diagnostics is None:
            diagnostics = {}
        gamma, iterations = solve_newton(target, seed, settings, jacobian=jacobian, diagnostics=diagnostics)
    except NonlinearSolveError as error:
        attempts.append(dict(solver='newton', status='failed', iterations=error.iterations,
                             message=str(error)))
    else:
        return accepted(gamma, iterations, dict(solver='newton'))
    for relaxation in settings.fallback_relaxations:
        gamma = seed.copy()
        attempt = dict(solver='picard', relaxation=relaxation)
        for update in range(settings.fallback_max_iterations+1):
            try:
                error = residual(gamma)
                maximum, limit = metrics(gamma, error)
            except (ValueError, FloatingPointError) as error:
                attempt.update(status='domain_error', iterations=update+1, message=str(error))
                break
            if maximum <= limit:
                return accepted(gamma, update+1, attempt)
            if update == settings.fallback_max_iterations:
                attempt.update(status='exhausted', iterations=update+1,
                               residual_m2_s=maximum, residual_limit_m2_s=limit)
                break
            with np.errstate(over='ignore', invalid='ignore'):
                gamma = gamma+relaxation*error
            if not np.isfinite(gamma).all():
                attempt.update(status='nonfinite', iterations=update+1)
                break
        attempts.append(attempt)
    raise HybridSolveError(attempts, diagnostics=diagnostics)


def section_solver_trace(trace_json, polar, alpha, residual, limit):
    """Add exact table-segment identifiers without altering the root."""
    if trace_json is None:
        return None
    trace = json.loads(trace_json)
    trace['final_residual_m2_s'] = residual
    trace['final_residual_limit_m2_s'] = limit
    tables = getattr(polar, 'polars', None)
    if tables is None and hasattr(polar, 'alpha_deg'):
        tables = [polar]*len(alpha)
    if tables is not None:
        trace['polar_segment_indices'] = [int(np.clip(np.searchsorted(p.alpha_deg, a,
            side='right')-1, 0, len(p.alpha_deg)-2)) for p,a in zip(tables, np.rad2deg(alpha))]
        trace['polar_outside_table'] = [bool(a < p.alpha_deg[0] or a > p.alpha_deg[-1])
                                        for p,a in zip(tables, np.rad2deg(alpha))]
        trace['polar_segment_convention'] = ('right-hand in-table segment; outside-table indices identify '
            'the terminal interval only; clamped coefficients have zero slope outside the table')
    return json.dumps(trace, sort_keys=True, allow_nan=False)


def validate_solver_trace(trace_json, residual, limit, panels):
    """Reject inconsistent optional metadata; legacy frames need no trace."""
    try:
        trace = json.loads(trace_json)
    except (TypeError, ValueError) as error:
        raise ValueError('invalid circulation solver trace JSON') from error
    if not isinstance(trace, dict):
        raise ValueError('circulation solver trace must be an object')
    if (trace.get('schema'), trace.get('policy')) not in (
            ('newton_picard_v1', 'newton_first_residual_root_v1'),
            ('newton_v1', 'newton_residual_root_v1')):
        raise ValueError('unknown circulation solver trace')
    if trace.get('jacobian', 'finite_difference') not in ('finite_difference','active_segment'):
        raise ValueError('unknown solver trace Jacobian')
    if 'updates' in trace and (type(trace['updates']) is not int or trace['updates'] < 0):
        raise ValueError('invalid Newton update count')
    if (trace['schema']=='newton_v1' or trace.get('jacobian')=='active_segment') and (
            'jacobian' not in trace or 'updates' not in trace):
        raise ValueError('explicit Newton trace lacks derivative/update provenance')
    attempts = trace.get('attempts', [])
    if (not isinstance(attempts,list) or not attempts or
            any(not isinstance(x,dict) for x in attempts) or attempts[-1].get('status') != 'accepted'):
        raise ValueError('solver trace has no final accepted attempt')
    def finite_number(value):
        return type(value) in (int,float) and np.isfinite(value)
    if attempts[0].get('solver') != 'newton':
        raise ValueError('solver trace must begin with Newton')
    if trace['schema']=='newton_v1' and len(attempts)!=1:
        raise ValueError('plain Newton trace cannot contain retries')
    for index, attempt in enumerate(attempts):
        if index and attempt.get('solver') != 'picard':
            raise ValueError('only Picard retries may follow Newton')
        if index < len(attempts)-1 and attempt.get('status') not in ('failed','domain_error','exhausted','nonfinite'):
            raise ValueError('invalid failed solver attempt status')
        iterations=attempt.get('iterations')
        if iterations is not None and (type(iterations) is not int or iterations < 1):
            raise ValueError('invalid solver attempt iteration count')
        if index:
            omega=attempt.get('relaxation')
            if not finite_number(omega) or not 0 < omega <= .1:
                raise ValueError('invalid Picard relaxation in trace')
    last = attempts[-1]
    if 'updates' in trace:
        iterations=last.get('iterations')
        if type(iterations) is not int:
            raise ValueError('invalid solver attempt iteration count')
        allowed=(iterations-1,iterations) if last.get('solver')=='newton' else (iterations-1,)
        if trace['updates'] not in allowed:
            raise ValueError('solver updates contradict iteration count')
    if (not finite_number(last.get('residual_m2_s')) or not finite_number(last.get('residual_limit_m2_s'))
            or not finite_number(trace.get('relaxation'))
            or trace['relaxation'] != last.get('relaxation',0.)
            or type(trace.get('fallback_used')) is not bool
            or type(last.get('iterations')) is not int):
        raise ValueError('invalid accepted solver attempt metadata')
    if (last.get('solver') not in ('newton','picard')
            or trace.get('selected_solver') != last['solver']
            or trace.get('fallback_used') != (last['solver']=='picard')
            or trace.get('final_residual_m2_s') != residual
            or trace.get('final_residual_limit_m2_s') != limit
            or last['residual_limit_m2_s'] <= 0
            or not 0 <= last.get('residual_m2_s', -1) <= last.get('residual_limit_m2_s', -2)):
        raise ValueError('solver trace contradicts accepted circulation frame')
    if any(x.get('status')=='accepted' for x in attempts[:-1]):
        raise ValueError('solver trace continued after accepting a root')
    if not finite_number(trace.get('delta_seed_linf_m2_s')) or trace['delta_seed_linf_m2_s'] < 0:
        raise ValueError('invalid solver seed distance')
    segments = trace.get('polar_segment_indices')
    if segments is not None and (not isinstance(segments,list) or len(segments)!=panels or
            any(type(i) is not int or i<0 for i in segments)):
        raise ValueError('invalid solver polar-segment metadata')
    outside=trace.get('polar_outside_table')
    if outside is not None and (not isinstance(outside,list) or len(outside)!=panels
                               or any(type(value) is not bool for value in outside)):
        raise ValueError('invalid solver polar-outside-table metadata')
    return trace
