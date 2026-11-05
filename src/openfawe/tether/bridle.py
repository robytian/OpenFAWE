# SPDX-License-Identifier: Apache-2.0
# Formulation transcribed from the Makani simulator, google/makani
# sim/models/rigid_bodies/wing.cc, Wing::CalcBridlePivot and Wing::CalcBridlePoint
# (commit 1d66923, Apache-2.0). Modifications (2026): NumPy, inertial components
# throughout, any body axes whose y axis is the span axis.
"""Bridle point of the kite: where the tether ends (G4 contract, section 3 item 4).

Model of the source: the two bridle legs meet the tether at the bridle point, which stays at the
fixed distance ``radius_m`` from the bridle pivot (the midpoint of the two attachments on the wing,
shifted along the span by the source's y offset) and swings about the body's span axis only. Its
direction from the pivot is the direction from the pivot to the last tether node with the span-wise
part removed. The tether force acts on the body at the bridle point, so it has no pitching moment
about the pivot line but does roll and yaw the body.

The velocity of the bridle point is the source's expression: the velocity of the pivot plus the part
of the last node's velocity relative to the pivot that lies in the body's x-z plane and is normal to
the pivot-to-point direction, scaled by radius / distance. It is the exact time derivative of the
position while the body translates or turns about its span axis; when the body rolls or yaws the
turning of the swing plane with the body is left out, as in the source. The velocity enters the tether
through the damping of the last segment and through the curvature rate of the last node (bending
damping), not through the position of anything.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bridle:
    """Bridle pivot in body axes and the pivot-to-point distance."""
    pivot_body_m: np.ndarray
    radius_m: float

    def __post_init__(self):
        pivot = np.array(self.pivot_body_m, dtype=float).reshape(3)
        if not (np.all(np.isfinite(pivot)) and np.isfinite(self.radius_m) and self.radius_m > 0.):
            raise ValueError("bridle pivot must be finite and the radius positive")
        pivot.setflags(write=False)
        object.__setattr__(self, "pivot_body_m", pivot)
        object.__setattr__(self, "radius_m", float(self.radius_m))

    def pivot(self, pose):
        """Inertial position and velocity of the pivot (a point of the body)."""
        position = pose.position_m+pose.rotation@self.pivot_body_m
        return position, pose.velocity_m_s+np.cross(pose.angular_velocity_rad_s, position-pose.position_m)

    def point(self, pose, node_position, node_velocity):
        """Inertial position and velocity of the bridle point for the given last tether node."""
        pivot, pivot_velocity = self.pivot(pose)
        span = pose.rotation[:, 1]                                  # body y axis in inertial components
        to_node = np.asarray(node_position, dtype=float)-pivot
        in_plane = to_node-np.dot(span, to_node)*span
        distance = max(float(np.linalg.norm(in_plane)), 1e-9)       # the source's guard
        norm = np.linalg.norm(in_plane)
        direction = in_plane/norm if norm > 0. else in_plane        # Vec3Normalize leaves a zero vector as it is
        position = pivot+self.radius_m*direction
        relative = np.asarray(node_velocity, dtype=float)-pivot_velocity
        relative = relative-np.dot(span, relative)*span             # the bridle cannot move along the span axis
        relative = relative-np.dot(direction, relative)*direction   # nor along its own length
        return position, pivot_velocity+self.radius_m/distance*relative


def load_bridle(path):
    """The bridle of an ``openfawe.rigid_body.v1`` case file."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "openfawe.rigid_body.v1" or data.get("frame") != "x_downstream_y_span_z_up":
        raise ValueError(f"{path}: not an openfawe.rigid_body.v1 file in the x_downstream_y_span_z_up frame")
    return Bridle(data["bridle"]["pivot_m"], data["bridle"]["radius_m"])
