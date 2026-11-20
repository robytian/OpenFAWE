"""Inflow fields of the studio (inertial frame: x along the wind, z up from the ground).

``Inflow`` is a callable ``(points_m, time_s) -> (n, 3)``, the form ``openfawe.aero.wake.ambient_velocity`` accepts. It is a
plain class (it pickles, so the worker process can build it from the spec):

- power-law shear ``U(z) = U_ref (z / z_ref) ** alpha`` with the height ratio floored at 0.1, as Makani's
  ``CalcWindShear`` (``openfawe.ground.makani_wind.calc_wind_shear``);
- an optional "1 - cosine" gust, uniform in space: ``A / 2 (1 - cos(2 pi (t - t0) / T))`` for ``t0 <= t <= t0 + T``, added
  along the wind, vertically or laterally. This is the studio's own simple disturbance, not a model of the library.

Without shear and gust the studio passes the plain wind vector instead (``uniform_wind``), as the library's own runs do.
"""
import numpy as np


class Inflow:
    def __init__(self, speed_m_s, *, shear_exponent=0.0, reference_height_m=1.0, gust="none", gust_amplitude_m_s=0.0,
                 gust_start_s=0.0, gust_duration_s=1.0):
        self.speed = float(speed_m_s)
        self.exponent = float(shear_exponent)
        self.reference = float(reference_height_m)
        self.gust = str(gust)
        self.amplitude = float(gust_amplitude_m_s)
        self.start = float(gust_start_s)
        self.duration = float(gust_duration_s)
        if self.gust not in ("none", "longitudinal", "vertical", "lateral"):
            raise ValueError(f"unknown gust {gust!r}")
        if self.exponent and not self.reference > 0:
            raise ValueError("wind shear needs a positive reference height")

    def gust_speed(self, time_s):
        if self.gust == "none":
            return 0.0
        phase = (float(time_s)-self.start)/self.duration
        if phase < 0.0 or phase > 1.0:
            return 0.0
        return 0.5*self.amplitude*(1.0-np.cos(2.0*np.pi*phase))

    def __call__(self, points_m, time_s):
        points = np.asarray(points_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("inflow points must have shape (n, 3)")
        out = np.zeros_like(points)
        if self.exponent:
            ratio = np.maximum(points[:, 2]/self.reference, 0.1)
            out[:, 0] = self.speed*ratio**self.exponent
        else:
            out[:, 0] = self.speed
        extra = self.gust_speed(time_s)
        if extra:
            out[:, {"longitudinal": 0, "lateral": 1, "vertical": 2}[self.gust]] += extra
        return out

    def describe(self):
        text = f"{self.speed:g} m/s along +x"
        if self.exponent:
            text += f", power-law shear alpha {self.exponent:g} about {self.reference:g} m"
        if self.gust != "none":
            text += (f", 1-cos {self.gust} gust {self.amplitude:g} m/s from {self.start:g} s for {self.duration:g} s "
                     "(uniform in space)")
        return text


def uniform_wind(speed_m_s):
    return np.array([float(speed_m_s), 0.0, 0.0])
