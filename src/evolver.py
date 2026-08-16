"""
Redirect stub — this module moved to core/evolution/evolver.py.
Kept for backward compatibility during the refactor transition.
"""
import warnings
warnings.warn(
    "src/evolver.py is deprecated — use core.evolution.evolver instead",
    DeprecationWarning, stacklevel=2
)
from core.evolution.evolver import Evolver, EvolutionResult  # noqa: F401
