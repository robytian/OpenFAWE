"""Prescribed motion of the aircraft (G4, contract docs/tasks/OF-0007-g4motion20261002).

``kinematics``: rigid-body motion input (samples, interpolation, consistency checks) and the analytic
motions of contract cases A1 (uniform translation) and A2 (steady circle); the others are not written yet. ``prescribed``: time marching of the vortex-wake aerodynamics along
a prescribed motion. ``rigid_body``: time propagation of one rigid body under given loads (G5 plant contract,
step 1a; the body alone, no coupling). ``plant``: the body and the tether advanced together over one aerodynamic step
with given held loads (step 1b part one; no aerodynamics in it). Nothing here is used unless it is called explicitly.
"""
