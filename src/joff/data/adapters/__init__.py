"""Dataset adapters and preset registry."""

from __future__ import annotations

from .base import CanonicalDataset, DatasetAdapter, DatasetCardAdapter, DatasetPreset, Segment
from .builtin import SyntheticCSTRFaultAdapter, SyntheticProcessAdapter
from .real_process import (
    CSTRFaultAdapter,
    HYFaultAdapter,
    HYQualityPredictionAdapter,
    MultiphaseFaultAdapter,
    NpyReconstructionAdapter,
    TEClassificationAdapter,
    TEFaultDiagnosisAdapter,
    TTSFaultDiagnosisAdapter,
    WPTMPCAdapter,
)
from .registry import DatasetRegistry


DATASET_REGISTRY = DatasetRegistry()
DATASET_REGISTRY.register(
    SyntheticCSTRFaultAdapter(),
    aliases=("cstr_fd", "cstr/fd", "CSTR/fd"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="te_fault_diagnosis",
        task_name="fault_diagnosis",
        description="Deterministic TE-style fault-diagnosis smoke dataset.",
        domain="tennessee_eastman",
    ),
    aliases=("te_fd", "te/fd", "TE/fd"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="te_classification",
        task_name="classification",
        description="Deterministic TE-style classification smoke dataset.",
        domain="tennessee_eastman",
    ),
    aliases=("te_cls", "te/cls", "TE/cls"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="cstr_closed_loop_fd",
        task_name="fault_diagnosis",
        description="Deterministic closed-loop CSTR fault-diagnosis smoke dataset.",
        domain="process_control",
    ),
    aliases=("cstr_fd_close", "cstr/fd_close", "CSTR/fd_close"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="tts_fault_diagnosis",
        task_name="fault_diagnosis",
        description="Deterministic TTS-style fault-diagnosis smoke dataset.",
        domain="process_control",
    ),
    aliases=("tts_fd", "tts/fd", "TTS/fd"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="hy_fault_diagnosis",
        task_name="fault_diagnosis",
        description="Deterministic HY-style fault-diagnosis smoke dataset.",
        domain="hydrocracking",
    ),
    aliases=("hy_fd", "hy/fd", "HY/fd"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="hy_quality_prediction",
        task_name="prediction",
        description="Deterministic HY quality-prediction smoke dataset.",
        domain="hydrocracking",
    ),
    aliases=("hy_prd", "HY_PRD"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="multiphase_fd",
        task_name="fault_diagnosis",
        description="Deterministic multiphase-flow fault-diagnosis smoke dataset.",
        domain="multiphase_flow",
    ),
    aliases=("Multiphase_Flow_Facility", "multiphase/fd"),
)
DATASET_REGISTRY.register(
    SyntheticProcessAdapter(
        name="wpt_mpc",
        task_name="mpc",
        description="Deterministic WPT-style MPC smoke dataset.",
        domain="model_predictive_control",
    ),
    aliases=("wpt", "WPT"),
)
DATASET_REGISTRY.register(
    TEFaultDiagnosisAdapter(),
    aliases=("te_fd", "te/fd", "TE/fd"),
    replace=True,
)
DATASET_REGISTRY.register(
    TEClassificationAdapter(),
    aliases=("te_cls", "te/cls", "TE/cls"),
    replace=True,
)
DATASET_REGISTRY.register(
    CSTRFaultAdapter(),
    aliases=("cstr_fd", "cstr/fd", "CSTR/fd"),
    replace=True,
)
DATASET_REGISTRY.register(
    CSTRFaultAdapter(
        name="cstr_closed_loop_fd",
        subdir="fd_close",
        feature_count=7,
        description="Closed-loop CSTR fault-diagnosis dataset.",
    ),
    aliases=("cstr_fd_close", "cstr/fd_close", "CSTR/fd_close"),
    replace=True,
)
DATASET_REGISTRY.register(
    TTSFaultDiagnosisAdapter(),
    aliases=("tts_fd", "tts/fd", "TTS/fd"),
    replace=True,
)
DATASET_REGISTRY.register(
    NpyReconstructionAdapter(
        name="tts_sui_fault_estimation",
        subdir="sui_fe",
        raw_folder="TTS/sui_fe",
        description="Three-tank-system SUI reconstruction/fault-estimation dataset.",
    ),
    aliases=("tts_sui_fe", "TTS/sui_fe"),
)
DATASET_REGISTRY.register(
    NpyReconstructionAdapter(
        name="ne_fault_estimation",
        subdir="sui_fe",
        raw_folder="NE/sui_fe",
        description="NE SUI reconstruction/fault-estimation dataset.",
    ),
    aliases=("ne_sui_fe", "NE/sui_fe"),
)
DATASET_REGISTRY.register(
    HYFaultAdapter(),
    aliases=("hy_fd", "hy/fd", "HY/fd"),
    replace=True,
)
DATASET_REGISTRY.register(
    HYQualityPredictionAdapter(),
    aliases=("hy_prd", "HY_PRD"),
    replace=True,
)
DATASET_REGISTRY.register(
    MultiphaseFaultAdapter(),
    aliases=("Multiphase_Flow_Facility", "multiphase/fd"),
    replace=True,
)
DATASET_REGISTRY.register(
    WPTMPCAdapter(),
    aliases=("wpt", "WPT"),
    replace=True,
)

LEGACY_SPECIAL_PRESETS = {
    "TE/fd": "te_fault_diagnosis",
    "TE/cls": "te_classification",
    "CSTR/fd": "cstr_fault_diagnosis",
    "CSTR/fd_close": "cstr_closed_loop_fd",
    "TTS/fd": "tts_fault_diagnosis",
    "HY/fd": "hy_fault_diagnosis",
    "HY_PRD": "hy_quality_prediction",
    "Multiphase_Flow_Facility": "multiphase_fd",
    "WPT": "wpt_mpc",
}


def list_dataset_presets() -> tuple[str, ...]:
    """List registered dataset preset names."""

    return DATASET_REGISTRY.list()


def register_dataset_adapter(
    adapter: DatasetAdapter,
    *,
    aliases: tuple[str, ...] | list[str] = (),
    replace: bool = False,
) -> None:
    """Register a dataset adapter in the package registry."""

    DATASET_REGISTRY.register(adapter, aliases=aliases, replace=replace)


__all__ = [
    "CanonicalDataset",
    "CSTRFaultAdapter",
    "DATASET_REGISTRY",
    "DatasetAdapter",
    "DatasetCardAdapter",
    "DatasetPreset",
    "DatasetRegistry",
    "HYFaultAdapter",
    "HYQualityPredictionAdapter",
    "LEGACY_SPECIAL_PRESETS",
    "MultiphaseFaultAdapter",
    "NpyReconstructionAdapter",
    "Segment",
    "SyntheticCSTRFaultAdapter",
    "SyntheticProcessAdapter",
    "TEClassificationAdapter",
    "TEFaultDiagnosisAdapter",
    "TTSFaultDiagnosisAdapter",
    "WPTMPCAdapter",
    "list_dataset_presets",
    "register_dataset_adapter",
]
