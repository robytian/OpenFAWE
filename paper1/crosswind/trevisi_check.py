"""Far-wake induced drag of the helical-wake model of Trevisi, Riboldi and Croce (2023b) for one crosswind loop, with the
pitch shortened by the near-wake induction of the model and with the pitch from the computed motion of the wake tips
(Sec. III D, Eq. (28)):

    lambda  = u0 / (U cos e)                       tangential speed over the wind component along the loop axis
    lambda0 = 1 / (1/lambda - C_L/(pi A))          model: axial advance reduced by the near-wake induction u0 C_L/(pi A)
    lambda0 = 2 pi R0 / h0,  h0 = (U cos e - w) T  computed: w the speed of the tips against the wind along the axis
    C_Di^f  = (1/4 pi) (C_L^2/(pi A)) kappa0^(pi/2) lambda0^(3/2),  kappa0 = b/(2 R0)
    lift loss = C_w (C_Di^f / C_L) / C_L           the induced incidence C_Di^f/C_L times the lift response C_w

The tip speed w is the slope, through the origin, of the axial displacement of each tip (dx cos e + dz sin e, from
``wake_figures.py`` ``tip_displacement``, lap by lap) against the number of loops, times D/T.

Usage: trevisi_check.py WAKE_FIGURES_JSON --cl C_L --radius R0 --speed u0 --wind U --elevation DEG --period T OUT.json
       [--span 25.7] [--aspect 20] [--cw 6.05]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def model(cl, radius, lam0, span, aspect, cw):
    k0 = span/(2.0*radius)
    cdi = (1.0/(4.0*math.pi))*(cl**2/(math.pi*aspect))*k0**(math.pi/2.0)*lam0**1.5
    return dict(kappa0=k0, lambda0=lam0, CDi_far=cdi, induced_incidence_deg=math.degrees(cdi/cl), lift_loss=cw*(cdi/cl)/cl)


def tip_speed(figures, elevation, period):
    d = json.loads(Path(figures).read_text(encoding="utf-8"))
    D = float(d["diameter_m"])
    ce, se = math.cos(math.radians(elevation)), math.sin(math.radians(elevation))
    out = {}
    for tip, laps in d["tip_displacement"].items():
        keys = sorted(laps)
        n = np.arange(1, len(keys)+1, dtype=float)
        axial = np.array([laps[k]["dx_over_D"]*ce+laps[k]["dz_over_D"]*se for k in keys])
        out[tip] = float(-(n@axial)/(n@n)*D/period)
    return out, D


def main(argv):
    p = argparse.ArgumentParser(prog="trevisi_check.py")
    p.add_argument("figures")
    p.add_argument("--cl", type=float, required=True)
    p.add_argument("--radius", type=float, required=True)
    p.add_argument("--speed", type=float, required=True)
    p.add_argument("--wind", type=float, required=True)
    p.add_argument("--elevation", type=float, required=True)
    p.add_argument("--period", type=float, required=True)
    p.add_argument("--span", type=float, default=25.7)
    p.add_argument("--aspect", type=float, default=20.0)
    p.add_argument("--cw", type=float, default=6.05)
    p.add_argument("out")
    a = p.parse_args(argv)
    axial = a.wind*math.cos(math.radians(a.elevation))
    lam = a.speed/axial
    shortening = a.speed*a.cl/(math.pi*a.aspect)
    with_model = model(a.cl, a.radius, 1.0/(1.0/lam-a.cl/(math.pi*a.aspect)), a.span, a.aspect, a.cw)
    tips, D = tip_speed(a.figures, a.elevation, a.period)
    w = float(np.mean(list(tips.values())))
    pitch = (axial-w)*a.period
    with_tips = model(a.cl, a.radius, 2.0*math.pi*a.radius/pitch, a.span, a.aspect, a.cw)
    result = dict(inputs=vars(a), axial_wind_m_s=axial, lambda_=lam, near_wake_shortening_m_s=shortening,
                  shortening_over_axial_wind=shortening/axial, model=with_model, tip_axial_speed_m_s=tips,
                  computed_pitch_m=pitch, computed_pitch_over_D=pitch/D, with_computed_pitch=with_tips)
    Path(a.out).write_text(json.dumps(result, indent=1)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
