"""Source-derived aerodynamic components with explicit qualification records."""

from .vortex import CoreModel, FilamentSingularity, segment_velocity, segments_velocity, ring_velocity
from .methods import AeroMethod,AeroBackend
from .aircraft import WingAssembly,PanelPolars,load_aircraft

__all__ = ["CoreModel", "FilamentSingularity", "segment_velocity", "segments_velocity", "ring_velocity",
           "AeroMethod", "AeroBackend", "WingAssembly", "PanelPolars", "load_aircraft"]
