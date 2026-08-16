"""
Redirect stub — this module moved to core/evolution/evolution_log.py.
Kept for backward compatibility during the refactor transition.
"""
import warnings
warnings.warn(
    "src/evolution_log.py is deprecated — use core.evolution.evolution_log instead",
    DeprecationWarning, stacklevel=2
)
from core.evolution.evolution_log import EvolutionLog  # noqa: F401
