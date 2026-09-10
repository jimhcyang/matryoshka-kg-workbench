"""Engine boundary types."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

from ..corpus import CorpusIndex
from ..progress import RunControl
from ..query import QueryPackage
from ..semantic_resources import SemanticResources
from ..decision_profile import DEFAULT_ENGINE_THRESHOLDS


@dataclass(frozen=True)
class EngineConfig:
    staged_top_docs: int = 150
    retrieval_mode: str = "staged"
    raw_use_mode: str = "validate"
    raw_validation_finalists: int = 20
    minimum_score: float = DEFAULT_ENGINE_THRESHOLDS["minimum_score"]
    minimum_margin: float = DEFAULT_ENGINE_THRESHOLDS["minimum_margin"]
    reconstruction_critical_floor: float = DEFAULT_ENGINE_THRESHOLDS["reconstruction_critical_floor"]
    reconstruction_identity_floor: float = DEFAULT_ENGINE_THRESHOLDS["reconstruction_identity_floor"]
    sentence_context_floor: float = DEFAULT_ENGINE_THRESHOLDS["sentence_context_floor"]
    sentence_event_floor: float = DEFAULT_ENGINE_THRESHOLDS["sentence_event_floor"]
    sentence_one_sided_context_floor: float = DEFAULT_ENGINE_THRESHOLDS["sentence_one_sided_context_floor"]
    sentence_one_sided_margin: float = DEFAULT_ENGINE_THRESHOLDS["sentence_one_sided_margin"]
    claim_direct_identity: float = DEFAULT_ENGINE_THRESHOLDS["claim_direct_identity"]
    claim_conflict_identity: float = DEFAULT_ENGINE_THRESHOLDS["claim_conflict_identity"]
    related_identity: float = DEFAULT_ENGINE_THRESHOLDS["related_identity"]
    contradiction_identity_floor: float = DEFAULT_ENGINE_THRESHOLDS["contradiction_identity_floor"]
    contradiction_anchor_floor: float = DEFAULT_ENGINE_THRESHOLDS["contradiction_anchor_floor"]
    value_redaction_title_containment_floor: float = DEFAULT_ENGINE_THRESHOLDS["value_redaction_title_containment_floor"]
    value_redaction_event_floor: float = DEFAULT_ENGINE_THRESHOLDS["value_redaction_event_floor"]
    sentence_local_anchor_floor: float = DEFAULT_ENGINE_THRESHOLDS["sentence_local_anchor_floor"]
    claim_background_lexical_floor: float = DEFAULT_ENGINE_THRESHOLDS["claim_background_lexical_floor"]
    claim_raw_validation_floor: float = DEFAULT_ENGINE_THRESHOLDS["claim_raw_validation_floor"]
    contradiction_participant_containment_floor: float = DEFAULT_ENGINE_THRESHOLDS["contradiction_participant_containment_floor"]
    contradiction_object_containment_floor: float = DEFAULT_ENGINE_THRESHOLDS["contradiction_object_containment_floor"]
    contradiction_route_relevance_floor: float = DEFAULT_ENGINE_THRESHOLDS["contradiction_route_relevance_floor"]
    contradiction_background_retrieval_floor: float = DEFAULT_ENGINE_THRESHOLDS["contradiction_background_retrieval_floor"]
    contradiction_raw_validation_floor: float = DEFAULT_ENGINE_THRESHOLDS["contradiction_raw_validation_floor"]
    reconstruction_literal_containment_floor: float = DEFAULT_ENGINE_THRESHOLDS["reconstruction_literal_containment_floor"]
    contradiction_propositions_per_class_per_document: int = 12
    contradiction_raw_sentences_per_class_per_document: int = 4
    max_reconstruction_docs: int = 10
    minimal_marginal_gain: float = 0.01


@dataclass
class EngineResult:
    task: str
    evaluation: pd.DataFrame
    artifacts: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineContext:
    corpus: CorpusIndex
    query: QueryPackage
    output_dir: Path
    control: RunControl
    config: EngineConfig
    semantic_resources: SemanticResources


class TaskEngine:
    task_name: str = ""

    def run(self, context: EngineContext) -> EngineResult:
        raise NotImplementedError
