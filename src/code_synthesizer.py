"""
Redirect stub — this module moved to core/evolution/code_synthesizer.py.
Kept for backward compatibility during the refactor transition.
"""
import warnings
warnings.warn(
    "src/code_synthesizer.py is deprecated — use core.evolution.code_synthesizer instead",
    DeprecationWarning, stacklevel=2
)
from core.evolution.code_synthesizer import CodeSynthesizer  # noqa: F401
