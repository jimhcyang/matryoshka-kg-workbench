"""Independent native task engines."""

from .claims import ClaimsEngine
from .contradiction import ContradictionEngine
from .reconstruction import ReconstructionEngine
from .sentence_redaction import SentenceRedactionEngine
from .value_redaction import ValueRedactionEngine

ENGINE_TYPES = {
    "reconstruction": ReconstructionEngine,
    "redaction": ValueRedactionEngine,
    "sentence_redaction": SentenceRedactionEngine,
    "claims": ClaimsEngine,
    "contradictions": ContradictionEngine,
}

__all__ = ["ENGINE_TYPES"]

