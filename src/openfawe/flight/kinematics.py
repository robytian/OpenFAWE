"""Rigid-body motion input for prescribed-motion runs (G4 contract, section 3 item 1).

A motion is a time series of the body origin position, the body attitude, the origin velocity and
the angular velocity, all in one inertial frame. Conventions:

- ``rotation`` maps body coordinates to inertial coordinates, ``x_inertial = rotation @ x_body +
  position`` (its columns are the body axes written in the inertial frame). The direction-cosine
  matrix "inertial to body" of the contract is its transpose.
- ``angular_velocity_rad_s`` is written in inertial components.
- The velocity of a material point of the body at inertial position ``p`` is
  ``velocity + angular_velocity x (p - position)``.

Between samples the position, the velocity and the angular velocity are interpolated linearly and
the attitude along the geodesic (the constant-rate rotation from one sample to the next, which is
what quaternion spherical interpolation gives). Runs that step exactly on the samples never
interpolate.
"""

from dataclasses import dataclass

import numpy as np

_EPS = np.finfo(float).eps


def _array(value, shape, name):
    try:
        result = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


def rotation_vector(rotation):
    """Rotation vector (axis times angle, radians) of a proper rotation with angle below pi."""
    rotation = np.asarray(rotation, dtype=float)
    cosine = np.clip(.5*(np.trace(rotation)-1.), -1., 1.)
    axis = np.array([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0], rotation[1, 0]-rotation[0, 1]])
    sine = .5*np.linalg.norm(axis)
    angle = np.arctan2(sine, cosine)
    if angle > np.pi-1e-6:
        raise ValueError("rotation between two samples is too close to a half turn to be unique")
    if sine < 1e-8:
        # Small angle: axis/(2 sin) -> axis/2 with the series of angle/sin(angle).
        return .5*axis*(1.+angle*angle/6.)
    return axis*(angle/(2.*sine))


def rotation_from_vector(vector):
    """Proper rotation of a rotation vector (Rodrigues formula)."""
    vector = np.asarray(vector, dtype=float)
    angle = np.linalg.norm(vector)
    skew = np.array([[0., -vector[2], vector[1]], [vector[2], 0., -vector[0]], [-vector[1], vector[0], 0.]])
    if angle < 1e-8:
        return np.eye(3)+skew+.5*skew@skew
    return np.eye(3)+np.sin(angle)/angle*skew+(1.-np.cos(angle))/(angle*angle)*skew@skew


@dataclass(frozen=True)
class Pose:
    """Position, attitude (body to inertial), origin velocity and angular velocity at one time."""
    time_s: float
    position_m: np.ndarray
    rotation: np.ndarray
    velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray

    def point_velocity(self, points):
        """Velocity of the body's material points at the inertial positions ``points`` (n,3)."""
        points = np.asarray(points, dtype=float)
        return self.velocity_m_s+np.cross(self.angular_velocity_rad_s, points-self.position_m)


@dataclass(frozen=True)
class RigidMotion:
    """Sampled rigid-body motion; see the module text for the conventions."""
    time_s: np.ndarray
    position_m: np.ndarray
    rotation: np.ndarray
    velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    acceleration_m_s2: np.ndarray | None = None
    angular_acceleration_rad_s2: np.ndarray | None = None
    description: str = "sampled motion"

    def __post_init__(self):
        time = np.array(self.time_s, dtype=float, copy=True)
        if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("motion times must be finite, strictly increasing, at least two")
        time.setflags(write=False)
        n = len(time)
        object.__setattr__(self, "time_s", time)
        for name in ("position_m", "velocity_m_s", "angular_velocity_rad_s"):
            object.__setattr__(self, name, _array(getattr(self, name), (n, 3), name))
        for name in ("acceleration_m_s2", "angular_acceleration_rad_s2"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _array(getattr(self, name), (n, 3), name))
        rotation = _array(self.rotation, (n, 3, 3), "rotation")
        identity = np.einsum("nij,nkj->nik", rotation, rotation)
        if (not np.allclose(identity, np.eye(3), rtol=0, atol=1e-12)
                or not np.allclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-12)):
            raise ValueError("every attitude sample must be a proper rotation")
        object.__setattr__(self, "rotation", rotation)

    def pose(self, time_s):
        """Pose at ``time_s``: the stored sample when the time is one, else the interpolation."""
        time = float(time_s)
        times = self.time_s
        tolerance = 8*_EPS*max(1., abs(time))
        if time < times[0]-tolerance or time > times[-1]+tolerance:
            raise ValueError("time outside the motion samples")
        index = int(np.searchsorted(times, time))
        for candidate in (index-1, index):
            if 0 <= candidate < len(times) and abs(times[candidate]-time) <= tolerance:
                return Pose(float(times[candidate]), self.position_m[candidate], self.rotation[candidate],
                            self.velocity_m_s[candidate], self.angular_velocity_rad_s[candidate])
        a, b = index-1, index
        s = (time-times[a])/(times[b]-times[a])
        def mix(values):
            return (1.-s)*values[a]+s*values[b]
        relative = rotation_vector(self.rotation[b]@self.rotation[a].T)
        return Pose(time, mix(self.position_m), rotation_from_vector(s*relative)@self.rotation[a],
                    mix(self.velocity_m_s), mix(self.angular_velocity_rad_s))

    def structural_velocity(self, points, time_s):
        """Body velocity at inertial points; the ``structural_velocity`` callable of the aerodynamics."""
        return self.pose(time_s).point_velocity(points)

    def consistency(self, attitude_scheme="interval"):
        """Self-consistency of the samples (criterion K3 of the contract; no force model involved).

        - velocity against the finite difference of the position: fourth-order central stencil on
          uniform samples (at least five), else the second-order non-uniform central difference;
        - angular velocity against the attitude, ``attitude_scheme``:
          ``interval`` (default): rotation vector of ``R[k+1] R[k]^T`` over the interval against the
          mean of the two stored angular velocities (exact for a constant angular velocity, second
          order otherwise; every interval is compared);
          ``fourth_order_central``: on uniform samples (at least five), the stored angular velocity at
          sample k against ``(8 phi_1 - phi_2) / (12 dt)`` with ``phi_m`` the rotation vector of
          ``R[k+m] R[k-m]^T`` (fourth order for a smooth attitude whose angular velocity changes
          direction, which the interval scheme sees only to second order; the first and last two
          samples are left out). With fewer than five samples or non-uniform steps it falls back to the
          interval scheme and says so in ``attitude_scheme``;
        - when accelerations are stored, the same differences of the velocity and of the angular
          velocity against them.
        Each entry gives the largest difference, the scale (largest stored magnitude), their ratio
        (``None`` when the scale is zero: such an entry has no relative measure and callers must
        judge ``max_abs`` themselves) and the number of samples or intervals compared. The central
        stencils leave out the first and last two samples (one for the second-order scheme). An
        error that alternates in sign from sample to sample is not seen by either check.
        """
        if attitude_scheme not in ("interval", "fourth_order_central"):
            raise ValueError("attitude_scheme is 'interval' or 'fourth_order_central'")
        time = self.time_s
        step = np.diff(time)
        uniform = len(time) >= 5 and np.allclose(step, step[0], rtol=1e-9, atol=0.)

        def derivative(values):
            if uniform:
                inner = (-values[4:]+8.*values[3:-1]-8.*values[1:-3]+values[:-4])/(12.*step[0])
                return inner, slice(2, -2)
            return np.gradient(values, time, axis=0)[1:-1], slice(1, -1)

        def entry(difference, reference):
            worst = float(np.max(np.linalg.norm(difference, axis=1))) if len(difference) else 0.
            scale = float(np.max(np.linalg.norm(reference, axis=1)))
            return dict(max_abs=worst, scale=scale, relative=None if scale == 0. else worst/scale,
                        compared=int(len(difference)))

        result = dict(scheme="fourth_order_central" if uniform else "second_order_central", samples=int(len(time)))
        fd, inner = derivative(self.position_m)
        result["velocity_vs_position"] = entry(fd-self.velocity_m_s[inner], self.velocity_m_s)
        if attitude_scheme == "fourth_order_central" and uniform:
            rotation = self.rotation
            phi1 = np.array([rotation_vector(rotation[k+1]@rotation[k-1].T) for k in range(2, len(time)-2)])
            phi2 = np.array([rotation_vector(rotation[k+2]@rotation[k-2].T) for k in range(2, len(time)-2)])
            estimate = (8.*phi1-phi2)/(12.*step[0])
            result["angular_velocity_vs_attitude"] = entry(estimate-self.angular_velocity_rad_s[2:-2], self.angular_velocity_rad_s)
            result["attitude_scheme"] = "fourth_order_central"
        else:
            increments = np.array([rotation_vector(self.rotation[k+1]@self.rotation[k].T)/step[k] for k in range(len(step))])
            mean = .5*(self.angular_velocity_rad_s[1:]+self.angular_velocity_rad_s[:-1])
            result["angular_velocity_vs_attitude"] = entry(increments-mean, self.angular_velocity_rad_s)
            result["attitude_scheme"] = "interval"
        if self.acceleration_m_s2 is not None:
            fd, inner = derivative(self.velocity_m_s)
            result["acceleration_vs_velocity"] = entry(fd-self.acceleration_m_s2[inner], self.acceleration_m_s2)
        if self.angular_acceleration_rad_s2 is not None:
            fd, inner = derivative(self.angular_velocity_rad_s)
            result["angular_acceleration_vs_angular_velocity"] = entry(
                fd-self.angular_acceleration_rad_s2[inner], self.angular_acceleration_rad_s2)
        return result


def relative_wind_body(pose, wind):
    """Air velocity relative to the body origin in body axes, with its speed, incidence and sideslip.

    Incidence ``atan2(w, u)`` and sideslip ``asin(v/V)`` of the relative wind ``(u, v, w)`` in the
    case-file body axes (x chordwise towards the trailing edge, y spanwise, z up), the same axes in
    which ``openfawe.aero.reference.inflow`` builds a wind from the two angles.
    """
    relative = pose.rotation.T@(np.asarray(wind, dtype=float)-pose.velocity_m_s)
    speed = float(np.linalg.norm(relative))
    if speed == 0.:
        return relative, 0., None, None
    return (relative, speed, float(np.degrees(np.arctan2(relative[2], relative[0]))),
            float(np.degrees(np.arcsin(np.clip(relative[1]/speed, -1., 1.)))))


def uniform_translation(times, velocity, *, rotation=None, origin=(0., 0., 0.)):
    """Constant velocity, constant attitude (case A1 of the contract); ``origin`` is the position at ``times[0]``."""
    times = np.asarray(times, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    origin = np.asarray(origin, dtype=float)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    n = len(times)
    return RigidMotion(times, origin+np.outer(times-times[0], velocity), np.broadcast_to(rotation, (n, 3, 3)),
                       np.broadcast_to(velocity, (n, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)),
                       description="uniform translation")


def circular_motion(times, radius_m, speed_m_s, wind_speed_m_s, alpha_deg, *, phase_rad=0.):
    """Steady circle in the plane normal to a uniform wind (case A2 of the contract).

    Inertial frame: the wind blows along +X with ``wind_speed_m_s``; the body origin moves on the
    circle of radius ``radius_m`` about the X axis in the Y-Z plane, ``position = R (0, cos, sin)`` of
    ``phase + Omega (t - times[0])`` with ``Omega = speed / R``, so the angular velocity is ``Omega``
    about +X (the sense of the loop is fixed; there is no mirrored option).
    Attitude: the span axis (body +y) points radially outwards; the chord axis (body x) lies along the
    relative wind at the origin turned by the incidence, so that the relative wind at the origin has
    the body components ``V (cos alpha, 0, sin alpha)`` with ``V = hypot(wind, speed)``: positive
    incidence is air coming from below (body +z is the lift side). Body z has the downwind
    component ``(wind sin alpha + speed cos alpha) / V``, positive for ``alpha > -atan(speed/wind)``.
    There is no bank, and the incidence is the one at the body origin only: panels further out see a
    higher speed and a smaller inflow angle.
    """
    times = np.asarray(times, dtype=float)
    radius, speed, wind = float(radius_m), float(speed_m_s), float(wind_speed_m_s)
    if radius <= 0. or speed <= 0. or wind < 0.:
        raise ValueError("radius and speed must be positive and the wind speed non-negative")
    rate = speed/radius
    angle = phase_rad+rate*(times-times[0])
    zero, one = np.zeros_like(angle), np.ones_like(angle)
    axis = np.column_stack([one, zero, zero])
    radial = np.column_stack([zero, np.cos(angle), np.sin(angle)])
    tangent = np.column_stack([zero, -np.sin(angle), np.cos(angle)])
    total = np.hypot(wind, speed)
    along = (wind*axis-speed*tangent)/total              # direction of the relative wind at the origin
    normal = (speed*axis+wind*tangent)/total             # normal to it with a downwind component
    a = np.radians(float(alpha_deg))
    body_x = np.cos(a)*along-np.sin(a)*normal
    body_z = np.sin(a)*along+np.cos(a)*normal
    rotation = np.stack([body_x, radial, body_z], axis=2)
    return RigidMotion(times, radius*radial, rotation, speed*tangent, rate*axis,
                       -speed*rate*radial, np.zeros((len(times), 3)),
                       description=f"circle R={radius:g} m, V={speed:g} m/s, wind {wind:g} m/s along the axis, alpha {float(alpha_deg):g} deg")


def trapezoid_profile(time, distance, ramp_s, cruise_s):
    """Angle, rate and acceleration of a rest-to-rest move of ``distance`` (signed) with a trapezoidal rate.

    The rate rises linearly for ``ramp_s``, stays constant for ``cruise_s`` and falls linearly for
    ``ramp_s``; ``time`` is measured from the start of the move and may lie outside it. At the two
    corner times the acceleration returned is that of the phase about to begin.
    """
    time = np.asarray(time, dtype=float)
    ramp, cruise = float(ramp_s), float(cruise_s)
    if ramp <= 0. or cruise < 0.:
        raise ValueError("ramp time must be positive and cruise time non-negative")
    peak = float(distance)/(ramp+cruise)
    slope = peak/ramp
    t = np.clip(time, 0., 2.*ramp+cruise)
    rising, falling = t < ramp, t >= ramp+cruise
    cruising = ~rising & ~falling
    left = 2.*ramp+cruise-t
    angle = np.where(rising, .5*slope*t*t, np.where(cruising, peak*(t-.5*ramp), float(distance)-.5*slope*left*left))
    rate = np.where(rising, slope*t, np.where(cruising, peak, slope*left))
    inside = (time >= 0.) & (time < 2.*ramp+cruise)
    acceleration = np.where(inside & rising, slope, np.where(inside & falling, -slope, 0.))
    return angle, np.where(inside, rate, 0.), acceleration


def pitch_manoeuvre(times, speed_m_s, alpha_deg, moves, *, origin=(0., 0., 0.)):
    """Constant-airspeed flight through still air with a prescribed incidence history (case A3 of the contract).

    Inertial frame: the body origin moves with ``speed_m_s`` along -X, so the relative wind at the
    origin is ``speed`` along +X. The body is turned about the inertial Y axis (which is the body y
    axis) by the incidence: body x = (cos alpha, 0, -sin alpha), body z = (sin alpha, 0, cos alpha), so
    the relative wind at the origin has the body components ``V (cos alpha, 0, sin alpha)`` and the
    angular velocity is ``d(alpha)/dt`` about +Y.

    ``alpha_deg`` is the incidence at ``times[0]``; ``moves`` is a sequence of
    ``(start_s, change_deg, ramp_s, cruise_s)``, each a rest-to-rest trapezoidal move
    (``trapezoid_profile``) beginning ``start_s`` after ``times[0]``; moves must not overlap and must
    lie inside the sampled time span (a truncated move would silently change the history). To keep
    the samples exactly consistent the corner times of every move should fall on samples. The angular
    acceleration is discontinuous at the corners and is therefore not stored; the linear acceleration
    (zero) is.
    """
    times = np.asarray(times, dtype=float)
    speed = float(speed_m_s)
    if speed <= 0.:
        raise ValueError("speed must be positive")
    elapsed = times-times[0]
    alpha = np.full(len(times), np.radians(float(alpha_deg)))
    rate = np.zeros(len(times))
    end_of_previous = -np.inf
    for start, change_deg, ramp, cruise in moves:
        if start < end_of_previous-1e-12*max(1., abs(end_of_previous)):       # back-to-back moves are allowed
            raise ValueError("moves must not overlap and must be given in time order")
        end_of_previous = start+2.*ramp+cruise
        if start < 0. or end_of_previous > elapsed[-1]*(1.+1e-12):
            raise ValueError("every move must lie inside the sampled time span")
        angle, speed_now, _ = trapezoid_profile(elapsed-start, np.radians(float(change_deg)), ramp, cruise)
        alpha = alpha+angle
        rate = rate+speed_now
    cos, sin, zero, one = np.cos(alpha), np.sin(alpha), np.zeros(len(times)), np.ones(len(times))
    rotation = np.stack([np.column_stack([cos, zero, -sin]), np.column_stack([zero, one, zero]),
                         np.column_stack([sin, zero, cos])], axis=2)
    velocity = np.broadcast_to([-speed, 0., 0.], (len(times), 3))
    position = np.asarray(origin, dtype=float)+np.outer(elapsed, [-speed, 0., 0.])
    return RigidMotion(times, position, rotation, velocity, np.column_stack([zero, rate, zero]),
                       np.zeros((len(times), 3)), None,
                       description=f"pitch manoeuvre at {speed:g} m/s from {float(alpha_deg):g} deg, {len(moves)} move(s)")


# ----- figure-eight on the tether sphere (OF-0009, case C2) ------------------------------------------------------

_GL_NODES = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
_GL_WEIGHTS = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])


class FigureEightPath:
    """Lissajous figure-eight on the sphere of radius ``radius_m`` about ``centre_m``, run at constant speed.

    Spherical coordinates about the centre: azimuth ``phi`` from the +X axis (the wind direction) in the
    horizontal plane, elevation ``beta`` above it, ``e(phi, beta) = (cos beta cos phi, cos beta sin phi, sin beta)``.
    The path is ``phi = phi_0 + A sin(theta)``, ``beta = beta_0 + B sin(2 theta)`` with the path parameter
    ``theta`` running over one turn per figure-eight: one azimuth oscillation and two elevation oscillations,
    which gives the lying eight. With ``B > 0`` the kite rises through the centre crossing and descends at the
    two sides (the down-loop pattern of pumping kites); ``B < 0`` is the up-loop pattern. The speed along the
    path is constant: ``theta(t)`` is the inverse of the arc length ``s(theta)``, tabulated by Gauss-Legendre
    quadrature of ``|dp/dtheta|`` on ``table_intervals`` intervals and inverted by Newton iteration, so that
    ``|dp/dt| = speed`` to round-off. The period is the path length over the speed.
    """

    def __init__(self, radius_m, speed_m_s, elevation_rad, azimuth_amplitude_rad, elevation_amplitude_rad, *,
                 azimuth_offset_rad=0., centre_m=(0., 0., 0.), table_intervals=16384):
        self.radius, self.speed = float(radius_m), float(speed_m_s)
        self.beta0, self.A, self.B = float(elevation_rad), float(azimuth_amplitude_rad), float(elevation_amplitude_rad)
        self.phi0 = float(azimuth_offset_rad)
        self.centre = np.array(centre_m, dtype=float).reshape(3)
        if self.radius <= 0. or self.speed <= 0. or self.A <= 0. or self.B == 0.:
            raise ValueError("radius, speed and the azimuth amplitude must be positive and the elevation amplitude nonzero")
        if not (-np.pi/2 < self.beta0-abs(self.B) and self.beta0+abs(self.B) < np.pi/2):
            raise ValueError("the elevation must stay strictly between -90 and 90 degrees")
        n = int(table_intervals)
        if n < 64:
            raise ValueError("table_intervals must be at least 64")
        self.theta_table = np.linspace(0., 2.*np.pi, n+1)
        self.s_table = np.concatenate([[0.], np.cumsum(self._arc(self.theta_table[:-1], self.theta_table[1:]))])
        self.length_m = float(self.s_table[-1])
        self.period_s = self.length_m/self.speed

    # --- the curve and its derivatives in the path parameter ---
    def _angles(self, theta):
        theta = np.asarray(theta, dtype=float)
        phi = self.phi0+self.A*np.sin(theta)
        beta = self.beta0+self.B*np.sin(2.*theta)
        d_phi, d_beta = self.A*np.cos(theta), 2.*self.B*np.cos(2.*theta)
        dd_phi, dd_beta = -self.A*np.sin(theta), -4.*self.B*np.sin(2.*theta)
        return phi, beta, d_phi, d_beta, dd_phi, dd_beta

    def position(self, theta):
        phi, beta = self._angles(theta)[:2]
        return self.centre+self.radius*np.stack([np.cos(beta)*np.cos(phi), np.cos(beta)*np.sin(phi), np.sin(beta)], axis=-1)

    def derivatives(self, theta):
        """``dp/dtheta`` and ``d2p/dtheta2`` (shape (..., 3))."""
        phi, beta, d_phi, d_beta, dd_phi, dd_beta = self._angles(theta)
        cb, sb, cp, sp = np.cos(beta), np.sin(beta), np.cos(phi), np.sin(phi)
        e_phi = np.stack([-cb*sp, cb*cp, np.zeros_like(cb)], axis=-1)
        e_beta = np.stack([-sb*cp, -sb*sp, cb], axis=-1)
        e_pp = np.stack([-cb*cp, -cb*sp, np.zeros_like(cb)], axis=-1)
        e_pb = np.stack([sb*sp, -sb*cp, np.zeros_like(cb)], axis=-1)
        e_bb = np.stack([-cb*cp, -cb*sp, -sb], axis=-1)
        first = self.radius*(e_phi*d_phi[..., None]+e_beta*d_beta[..., None])
        second = self.radius*(e_pp*(d_phi*d_phi)[..., None]+2.*e_pb*(d_phi*d_beta)[..., None]+e_bb*(d_beta*d_beta)[..., None]
                              + e_phi*dd_phi[..., None]+e_beta*dd_beta[..., None])
        return first, second

    def _rate(self, theta):
        return np.linalg.norm(self.derivatives(theta)[0], axis=-1)

    def _arc(self, a, b):
        """Arc length from ``a`` to ``b`` (arrays), four-point Gauss-Legendre on each interval."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        half, mid = .5*(b-a), .5*(a+b)
        total = np.zeros(np.broadcast(a, b).shape)
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
            total = total+weight*self._rate(mid+half*node)
        return half*total

    def theta(self, time_s, t0=0.):
        """Path parameter at ``time_s`` (array): the inverse of the arc length at constant speed."""
        target = np.mod(self.speed*(np.asarray(time_s, dtype=float)-t0), self.length_m)
        theta = np.interp(target, self.s_table, self.theta_table)
        for _ in range(4):
            j = np.clip(np.searchsorted(self.theta_table, theta, side="right")-1, 0, len(self.theta_table)-2)
            s = self.s_table[j]+self._arc(self.theta_table[j], theta)
            theta = theta-(s-target)/self._rate(theta)
        return theta

    def state(self, time_s, t0=0.):
        """Position, velocity and acceleration of the point at ``time_s`` (arrays of shape (n, 3))."""
        theta = self.theta(time_s, t0)
        first, second = self.derivatives(theta)
        rate = np.linalg.norm(first, axis=-1)
        theta_dot = self.speed/rate
        theta_ddot = -self.speed*np.einsum("ij,ij->i", first, second)/rate**3*theta_dot
        position = self.position(theta)
        velocity = first*theta_dot[:, None]
        acceleration = second*(theta_dot*theta_dot)[:, None]+first*theta_ddot[:, None]
        return position, velocity, acceleration


class SphericalCirclePath:
    """Circle on the sphere of radius ``radius_m`` about ``centre_m``, run at constant speed (OF-0009 case C2c/C2e).

    The circle is the set of points at the angular distance ``angular_radius_rad`` from the centre direction
    ``c = e(azimuth_offset, elevation)`` (the conventions of ``FigureEightPath``): ``p = centre + R (cos rho c +
    sin rho (cos theta e1 + sin theta e2))`` with ``e1`` the horizontal unit vector to the right of ``c`` (``c x Z``
    normalised) and ``e2 = c x e1`` (upwards), so that ``theta = 0`` is the rightmost point and the path runs
    upwards from there (``sense = +1``; ``-1`` reverses it). The speed along the path is ``speed_m_s``, the period
    ``2 pi R sin rho / speed``. The interface (``position``, ``derivatives``, ``theta``, ``state``, ``period_s``,
    ``length_m``) is that of ``FigureEightPath``.
    """

    def __init__(self, radius_m, speed_m_s, elevation_rad, angular_radius_rad, *, azimuth_offset_rad=0., centre_m=(0., 0., 0.), sense=1):
        self.radius, self.speed = float(radius_m), float(speed_m_s)
        self.beta0, self.rho, self.phi0 = float(elevation_rad), float(angular_radius_rad), float(azimuth_offset_rad)
        self.sense = 1 if sense >= 0 else -1
        self.centre = np.array(centre_m, dtype=float).reshape(3)
        if self.radius <= 0. or self.speed <= 0. or not 0. < self.rho < np.pi/2:
            raise ValueError("radius and speed must be positive and the angular radius between 0 and 90 degrees")
        if not -np.pi/2 < self.beta0 < np.pi/2:
            raise ValueError("the elevation must lie strictly between -90 and 90 degrees")
        c = np.array([np.cos(self.beta0)*np.cos(self.phi0), np.cos(self.beta0)*np.sin(self.phi0), np.sin(self.beta0)])
        e1 = np.cross(c, [0., 0., 1.])
        e1 = e1/np.linalg.norm(e1)
        self.axis, self.e1, self.e2 = c, e1, np.cross(c, e1)
        self.length_m = 2.*np.pi*self.radius*np.sin(self.rho)
        self.period_s = self.length_m/self.speed

    def position(self, theta):
        theta = np.asarray(theta, dtype=float)
        ring = np.cos(theta)[..., None]*self.e1+np.sin(theta)[..., None]*self.e2
        return self.centre+self.radius*(np.cos(self.rho)*self.axis+np.sin(self.rho)*ring)

    def derivatives(self, theta):
        theta = np.asarray(theta, dtype=float)
        first = self.radius*np.sin(self.rho)*(-np.sin(theta)[..., None]*self.e1+np.cos(theta)[..., None]*self.e2)
        second = -self.radius*np.sin(self.rho)*(np.cos(theta)[..., None]*self.e1+np.sin(theta)[..., None]*self.e2)
        return first, second

    def theta(self, time_s, t0=0.):
        return self.sense*self.speed*(np.asarray(time_s, dtype=float)-t0)/(self.radius*np.sin(self.rho))

    def state(self, time_s, t0=0.):
        theta = self.theta(time_s, t0)
        first, second = self.derivatives(theta)
        rate = self.sense*self.speed/(self.radius*np.sin(self.rho))
        return self.position(theta), first*rate, second*rate*rate


def _attitude_from_relative_wind(position, velocity, centre, wind, alpha_rad):
    """Body axes of a tethered kite: span normal to the relative wind and to the radial direction, chord along
    the relative wind turned by the incidence, lift side towards the outward radial direction; rows (n, 3, 3)."""
    relative = wind[None, :]-velocity                       # air velocity relative to the body origin
    speed = np.linalg.norm(relative, axis=1)
    if np.any(speed < 1e-9):
        raise ValueError("the relative wind vanishes at a sample: the attitude is undefined")
    along = relative/speed[:, None]
    radial = position-centre[None, :]
    radial = radial/np.linalg.norm(radial, axis=1)[:, None]
    normal = radial-np.einsum("ij,ij->i", radial, along)[:, None]*along
    size = np.linalg.norm(normal, axis=1)
    if np.any(size < 1e-9):
        raise ValueError("the relative wind is radial at a sample: the attitude is undefined")
    normal = normal/size[:, None]
    body_x = np.cos(alpha_rad)*along-np.sin(alpha_rad)*normal
    body_z = np.sin(alpha_rad)*along+np.cos(alpha_rad)*normal
    body_y = np.cross(body_z, body_x)
    return np.stack([body_x, body_y, body_z], axis=2)


def spherical_path_motion(times, path, wind_speed_m_s, alpha_deg, *, t0_s=None, difference_step_s=1e-3, description=None):
    """Rigid motion of a tethered kite along a constant-speed path on its tether sphere (``FigureEightPath`` or
    ``SphericalCirclePath``), with the attitude rule of ``_attitude_from_relative_wind``: the span axis (body y) is normal
    to the relative wind at the body origin and to the radial direction from the sphere centre; the chord axis (body x)
    lies along the relative wind turned by the incidence ``alpha_deg`` about the span axis, so that the relative wind at
    the origin has the body components ``V (cos alpha, 0, sin alpha)`` with zero sideslip (the convention of
    ``circular_motion``); the lift side (body z) has a positive component along the outward radial direction, towards
    which the tether pulls. Inertial frame: wind along +X with ``wind_speed_m_s``, Z up, the sphere centre is the ground
    attachment point. The velocity and the acceleration are those of the path (analytic in the path parameter); the
    angular velocity is the fourth-order central difference (step ``difference_step_s``) of the attitude, which
    ``RigidMotion.consistency('fourth_order_central')`` reproduces to the order of the sampling; the angular acceleration
    is not stored (the aerodynamics does not use it and its sampled difference is at the 1e-6 limit of K3 in the turns
    of a figure-eight at a 0.025 s step), as for the Makani loop of A5. The motion starts at ``theta = 0`` at ``times[0]``
    unless ``t0_s`` says otherwise. Returns ``(motion, path)``.
    """
    times = np.asarray(times, dtype=float)
    wind = np.array([float(wind_speed_m_s), 0., 0.])
    if wind[0] < 0.:
        raise ValueError("the wind speed must be non-negative")
    alpha = np.radians(float(alpha_deg))
    t0 = times[0] if t0_s is None else float(t0_s)
    h = float(difference_step_s)
    if h <= 0.:
        raise ValueError("difference_step_s must be positive")

    def attitude(at):
        position, velocity, _ = path.state(at, t0)
        return _attitude_from_relative_wind(position, velocity, path.centre, wind, alpha)

    position, velocity, acceleration = path.state(times, t0)
    rotation = attitude(times)
    rotations = {m: attitude(times+m*h) for m in (-2, -1, 1, 2)}
    phi1 = np.array([rotation_vector(b@a.T) for a, b in zip(rotations[-1], rotations[1])])
    phi2 = np.array([rotation_vector(b@a.T) for a, b in zip(rotations[-2], rotations[2])])
    omega = (8.*phi1-phi2)/(12.*h)
    if description is None:
        description = (f"{type(path).__name__} R={path.radius:g} m about {path.centre.tolist()}, speed {path.speed:g} m/s, wind {wind[0]:g} m/s "
                       f"along +X, incidence {float(alpha_deg):g} deg, period {path.period_s:.4f} s, path length {path.length_m:.2f} m")
    motion = RigidMotion(times, position, rotation, velocity, omega, acceleration, None, description=description)
    return motion, path


def figure_eight_motion(times, radius_m, speed_m_s, wind_speed_m_s, alpha_deg, *, elevation_deg, azimuth_amplitude_deg,
                        elevation_amplitude_deg, azimuth_offset_deg=0., centre_m=(0., 0., 0.), t0_s=None,
                        difference_step_s=1e-3, table_intervals=16384):
    """Figure-eight crosswind flight of a tethered kite at constant speed (case C2 of the OF-0009 contract).

    The path is ``FigureEightPath`` (azimuth ``azimuth_offset + azimuth_amplitude sin(theta)``, elevation
    ``elevation + elevation_amplitude sin(2 theta)`` on the sphere of radius ``radius_m`` about ``centre_m``, constant
    speed ``speed_m_s``; a positive elevation amplitude gives the down-loop pattern) and the motion is
    ``spherical_path_motion`` with its attitude rule. Returns ``(motion, path)``.
    """
    path = FigureEightPath(radius_m, speed_m_s, np.radians(elevation_deg), np.radians(azimuth_amplitude_deg),
                           np.radians(elevation_amplitude_deg), azimuth_offset_rad=np.radians(azimuth_offset_deg),
                           centre_m=centre_m, table_intervals=table_intervals)
    description = (f"figure-eight on the sphere R={path.radius:g} m about {path.centre.tolist()}, speed {path.speed:g} m/s, "
                   f"wind {float(wind_speed_m_s):g} m/s along +X, elevation {float(elevation_deg):g} deg +/- {float(elevation_amplitude_deg):g} deg, "
                   f"azimuth {float(azimuth_offset_deg):g} deg +/- {float(azimuth_amplitude_deg):g} deg, incidence {float(alpha_deg):g} deg, "
                   f"period {path.period_s:.4f} s, path length {path.length_m:.2f} m")
    return spherical_path_motion(times, path, wind_speed_m_s, alpha_deg, t0_s=t0_s, difference_step_s=difference_step_s, description=description)


def spherical_circle_motion(times, radius_m, speed_m_s, wind_speed_m_s, alpha_deg, *, elevation_deg, angular_radius_deg,
                            azimuth_offset_deg=0., centre_m=(0., 0., 0.), sense=1, t0_s=None, difference_step_s=1e-3):
    """Circular crosswind flight of a tethered kite at constant speed on its tether sphere (cases C2c and C2e of OF-0009).

    The path is ``SphericalCirclePath`` (centre direction at ``elevation_deg`` and ``azimuth_offset_deg``, angular
    radius ``angular_radius_deg``, ``sense`` +1 upwards from the rightmost point) and the motion is
    ``spherical_path_motion`` with its attitude rule (incidence ``alpha_deg`` at the origin, zero sideslip, span axis
    normal to the radial direction). Unlike ``circular_motion`` (case A2: the circle about the wind axis with the span
    radial, no sphere) and the Makani nominal loop (case A5: the controller's attitude law), this is the same
    kinematic rule as the figure-eight, for the comparison of the two patterns. Returns ``(motion, path)``.
    """
    path = SphericalCirclePath(radius_m, speed_m_s, np.radians(elevation_deg), np.radians(angular_radius_deg),
                               azimuth_offset_rad=np.radians(azimuth_offset_deg), centre_m=centre_m, sense=sense)
    description = (f"circle on the sphere R={path.radius:g} m about {path.centre.tolist()}, speed {path.speed:g} m/s, wind {float(wind_speed_m_s):g} m/s "
                   f"along +X, centre at elevation {float(elevation_deg):g} deg and azimuth {float(azimuth_offset_deg):g} deg, angular radius "
                   f"{float(angular_radius_deg):g} deg (path radius {path.radius*np.sin(path.rho):.2f} m), sense {path.sense:+d}, incidence {float(alpha_deg):g} deg, "
                   f"period {path.period_s:.4f} s")
    return spherical_path_motion(times, path, wind_speed_m_s, alpha_deg, t0_s=t0_s, difference_step_s=difference_step_s, description=description)
