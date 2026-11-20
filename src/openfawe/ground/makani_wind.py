"""Makani's wind shear (borrowed part, transcribed; opt-in, used by nothing by default).

Source: Makani snapshot 1d66923, ``sim/physics/wind.cc:37-44`` ``CalcWindShear`` (Apache License 2.0, Makani
Technologies LLC): a power law referenced to the ground, the height ratio floored at 0.1. Power-energy contract (plan
3.9b): compared bit for bit in E2a; used by ``ground.wind_shear`` (addenda 5, 7 and 8).
Makani's defaults (``config/common/sim/phys_sim.py``): exponent 0, reference height 21 m above ground (the anemometer).
Bit-identical to the source for finite inputs and a non-zero reference height (E2's development check); for NaN Python's
``max`` and C's ``fmax`` differ, and a zero reference height raises here where C divides (review B-R03).
"""
import math


def calc_wind_shear(height_agl, ref_height_agl, ref_wind_speed, wind_shear_exponent):
    """``CalcWindShear``: ``ref_wind_speed * pow(fmax(height_agl / ref_height_agl, 0.1), wind_shear_exponent)``."""
    return ref_wind_speed*math.pow(max(height_agl/ref_height_agl, 0.1), wind_shear_exponent)
