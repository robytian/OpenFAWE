"""Platform model P2 (OF-0010 contract revision 3 with its addenda 1-4; decision 0053 addendum 5): the finite core between
lifting surfaces that AVL uses, for the M600's surfaces solved in one vortex system.

The rule (AVL, M. Drela and H. Youngren, MIT; Makani's copy ``third_party/avl``: ``src/aic.f`` 92-101, ``src/avl.f`` 670,
``avl_doc.txt`` 954-963): a vortex of one surface acts on a control point of another surface through a finite core of
radius r_c = max(VRCORE c, 2 VRCORE ds), VRCORE = 0.25, c the chord of the vortex's strip and ds its width in the y-z
plane. AVL's finite-segment formula (``aic.f`` 600-611) is the exact integral of dl x r / (r^2 + r_c^2)^(3/2) along the
segment, which is the repository's ``CoreModel.ROSENHEAD_MOORE``; nothing of AVL's (GPL) code is used.

Which pairs (the owner's option (b), decision 0053 addendum 5): only pairs of different *classes*, a class being the
main lifting surfaces together (wing, horizontal and vertical tail: everything among them stays as in P1) or one pylon.
The precedent for using the core between joined surfaces is Makani's own AVL model of the M600 (all surfaces separate,
no COMPONENT); AVL's documentation asks joined surfaces to be one COMPONENT without the core and does not treat a
cruciform junction - so the choice is the contract's (revision 3 addendum 3, item 6).

Choices of the contract, not AVL's: the radius of an element is max(r_c, its material core); a streamwise wake line
shared by two strips takes the larger of the two strips' r_c; free wake rows carry the rule too (AVL's trailing legs
are semi-infinite with a different regularised form, ``aic.f`` 616-641); the wake's convection and the rotor rings are
unchanged. c is the mean of the strip's two node chords and ds the width of its bound segment in the body y-z plane,
both from the body-axis geometry given once (they do not follow the attitude or the elevator turn).

Nothing in the package imports this module; with ``surfaces=None`` in ``aero.vsm_wake`` every result is unchanged.
"""
from dataclasses import dataclass, field

import numpy as np

from .vortex import CoreModel

VRCORE = 0.25
CROSS_MODEL = CoreModel.ROSENHEAD_MOORE


def _owned(values, dtype):
    array = np.array(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SurfaceCores:
    """Class of every panel and node, and the AVL radius of every strip, for one assembly topology."""
    panel_class: np.ndarray
    node_class: np.ndarray
    panel_radius: np.ndarray
    node_radius: np.ndarray
    panel_edges: np.ndarray
    names: tuple
    surface_classes: tuple
    vrcore: float = VRCORE
    span_panel: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        for name, dtype in (("panel_class", np.int64), ("node_class", np.int64), ("panel_radius", float),
                            ("node_radius", float), ("panel_edges", np.int64)):
            object.__setattr__(self, name, _owned(getattr(self, name), dtype))
        if self.panel_edges.ndim != 2 or self.panel_edges.shape[1] != 2:
            raise ValueError("panel edges must be (panels, 2)")
        if len(self.panel_class) != len(self.panel_edges) or len(self.panel_radius) != len(self.panel_edges):
            raise ValueError("one class and one radius per panel")
        if len(self.node_radius) != len(self.node_class):
            raise ValueError("one radius per node")
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "surface_classes", tuple(int(c) for c in self.surface_classes))
        if len(self.names) != len(self.surface_classes):
            raise ValueError("one class per surface name")
        if not (np.all(np.isfinite(self.panel_radius)) and np.all(self.panel_radius > 0)):
            raise ValueError("strip radii must be finite and positive")
        width = len(self.node_class)
        lookup = np.full((width, width), -1, dtype=np.int64)
        for j, (a, b) in enumerate(self.panel_edges):
            if self.node_class[a] != self.panel_class[j] or self.node_class[b] != self.panel_class[j]:
                raise ValueError("a panel and its nodes must have one class")
            lookup[a, b] = lookup[b, a] = j
        lookup.setflags(write=False)
        object.__setattr__(self, "span_panel", lookup)

    @classmethod
    def from_assembly(cls, assembly, surface_classes, *, vrcore=VRCORE):
        """``assembly`` (a WingAssembly, body axes, the geometry the radii are taken from once), ``surface_classes`` one
        class number per surface in ``assembly.wings`` order (equal numbers: one class)."""
        surface_classes = tuple(int(c) for c in surface_classes)
        if len(surface_classes) != len(assembly.wings):
            raise ValueError("one class per surface")
        panel_class, node_class, radius, node_radius = [], [], [], []
        for wing, c in zip(assembly.wings, surface_classes):
            le, te = np.asarray(wing.leading_edge, dtype=float), np.asarray(wing.trailing_edge, dtype=float)
            chord_nodes = np.linalg.norm(te-le, axis=1)
            chord = 0.5*(chord_nodes[:-1]+chord_nodes[1:])
            bound = np.asarray(wing.bound, dtype=float)
            width = np.hypot(np.diff(bound[:, 1]), np.diff(bound[:, 2]))
            r = np.maximum(vrcore*chord, 2.0*vrcore*width)
            panel_class.extend([c]*wing.panels)
            node_class.extend([c]*wing.node_count)
            radius.extend(r.tolist())
            per_node = np.zeros(wing.node_count)
            per_node[:-1] = r
            per_node[1:] = np.maximum(per_node[1:], r)
            node_radius.extend(per_node.tolist())
        return cls(np.array(panel_class), np.array(node_class), np.array(radius), np.array(node_radius),
                   np.asarray(assembly.panel_edges), tuple(assembly.names), surface_classes, vrcore=float(vrcore))

    @property
    def classes(self):
        return tuple(int(c) for c in np.unique(self.panel_class))

    def check(self, wing):
        """Raise unless ``wing`` has this topology."""
        if wing.node_count != len(self.node_class) or not np.array_equal(np.asarray(wing.panel_edges), self.panel_edges):
            raise ValueError("the surfaces' classes do not belong to this wing's topology")

    def edge_class(self, edges, width):
        """Class of each edge of a row-major network (node index = row*width + node)."""
        edges = np.asarray(edges, dtype=np.int64)
        return self.node_class[edges[:, 0] % width]

    def edge_radius(self, edges, width):
        """AVL radius of each edge: a spanwise edge its strip's, a streamwise edge the larger of its node's strips'."""
        edges = np.asarray(edges, dtype=np.int64)
        ra, na = np.divmod(edges[:, 0], width)
        rb, nb = np.divmod(edges[:, 1], width)
        out = np.empty(len(edges))
        span = ra == rb
        panels = self.span_panel[na[span], nb[span]]
        if np.any(panels < 0):
            raise ValueError("a spanwise edge that is not a panel's")
        out[span] = self.panel_radius[panels]
        stream = ~span
        if np.any(na[stream] != nb[stream]) or np.any(np.abs(rb[stream]-ra[stream]) != 1):
            raise ValueError("an edge that is neither spanwise nor streamwise")
        out[stream] = self.node_radius[na[stream]]
        return out

    def settings(self):
        return dict(rule="AVL inter-surface finite core (aic.f 92-101, avl.f 670), Rosenhead-Moore kernel, only between "
                         "different classes (option (b), decision 0053 addendum 5)",
                    vrcore=self.vrcore, model="ROSENHEAD_MOORE", radius="max(VRCORE c, 2 VRCORE ds, material core)",
                    classes={str(c): [n for n, k in zip(self.names, self.surface_classes) if k == c] for c in self.classes},
                    strip_radius_m=dict(min=float(self.panel_radius.min()), max=float(self.panel_radius.max())))
