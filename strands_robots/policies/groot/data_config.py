#!/usr/bin/env python3
"""
GR00T Data Configuration

Provides data configurations for GR00T policy inference.
Uses Isaac-GR00T's native types when available, falls back to embedded implementations.

When `gr00t` is installed:
  - Uses `gr00t.data.types.ModalityConfig` directly
  - Uses `gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS` for registry
  - Full compatibility with Isaac-GR00T ecosystem

When `gr00t` is NOT installed:
  - Uses lightweight embedded ModalityConfig dataclass
  - Provides built-in configs for common robots (SO-100, GR-1, G1, Panda)

SPDX-License-Identifier: Apache-2.0
"""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to use Isaac-GR00T native types; fall back to lightweight embedded ones
# ---------------------------------------------------------------------------
_USING_GROOT = False

try:
    from gr00t.data.types import ModalityConfig  # noqa: F401

    _USING_GROOT = True
    logger.debug("Using gr00t.data.types.ModalityConfig (Isaac-GR00T installed)")
except ImportError:
    logger.debug("Isaac-GR00T not installed – using embedded ModalityConfig")

    @dataclass
    class ModalityConfig:
        """Lightweight ModalityConfig (embedded fallback).

        Compatible subset of ``gr00t.data.types.ModalityConfig``.
        """

        delta_indices: List[int]
        modality_keys: List[str]

        def model_dump_json(self) -> str:
            import json

            return json.dumps({"delta_indices": self.delta_indices, "modality_keys": self.modality_keys})


# ---------------------------------------------------------------------------
# Base data config
# ---------------------------------------------------------------------------


@dataclass
class BaseDataConfig(ABC):
    """Abstract base for GR00T data configurations.

    Subclasses define camera, state, action, and language keys for a specific
    robot embodiment.
    """

    video_keys: List[str]
    state_keys: List[str]
    action_keys: List[str]
    language_keys: List[str]
    observation_indices: List[int]
    action_indices: List[int]

    def modality_config(self) -> Dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }


# ---------------------------------------------------------------------------
# Concrete embodiment configs
# ---------------------------------------------------------------------------

_DEFAULT_OBS = [0]
_DEFAULT_ACT = list(range(16))


@dataclass
class So100DataConfig(BaseDataConfig):
    """SO-100 single camera."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.webcam"]
        self.state_keys = self.state_keys or ["state.single_arm", "state.gripper"]
        self.action_keys = self.action_keys or ["action.single_arm", "action.gripper"]
        self.language_keys = self.language_keys or ["annotation.human.task_description"]
        self.observation_indices = self.observation_indices or _DEFAULT_OBS
        self.action_indices = self.action_indices or _DEFAULT_ACT


@dataclass
class So100DualCamDataConfig(BaseDataConfig):
    """SO-100 dual camera."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.front", "video.wrist"]
        self.state_keys = self.state_keys or ["state.single_arm", "state.gripper"]
        self.action_keys = self.action_keys or ["action.single_arm", "action.gripper"]
        self.language_keys = self.language_keys or ["annotation.human.task_description"]
        self.observation_indices = self.observation_indices or _DEFAULT_OBS
        self.action_indices = self.action_indices or _DEFAULT_ACT


@dataclass
class So100QuadCamDataConfig(BaseDataConfig):
    """SO-100 quad camera."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.front", "video.wrist", "video.top", "video.side"]
        self.state_keys = self.state_keys or ["state.single_arm", "state.gripper"]
        self.action_keys = self.action_keys or ["action.single_arm", "action.gripper"]
        self.language_keys = self.language_keys or ["annotation.human.task_description"]
        self.observation_indices = self.observation_indices or _DEFAULT_OBS
        self.action_indices = self.action_indices or _DEFAULT_ACT


@dataclass
class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    """Fourier GR-1 arms only."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.ego_view"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or _DEFAULT_OBS
        self.action_indices = self.action_indices or _DEFAULT_ACT


@dataclass
class BimanualPandaGripperDataConfig(BaseDataConfig):
    """Bimanual Panda gripper."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"]
        self.state_keys = self.state_keys or [
            "state.right_arm_eef_pos",
            "state.right_arm_eef_quat",
            "state.right_gripper_qpos",
            "state.left_arm_eef_pos",
            "state.left_arm_eef_quat",
            "state.left_gripper_qpos",
        ]
        self.action_keys = self.action_keys or [
            "action.right_arm_eef_pos",
            "action.right_arm_eef_rot",
            "action.right_gripper_close",
            "action.left_arm_eef_pos",
            "action.left_arm_eef_rot",
            "action.left_gripper_close",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or _DEFAULT_OBS
        self.action_indices = self.action_indices or _DEFAULT_ACT


@dataclass
class UnitreeG1DataConfig(BaseDataConfig):
    """Unitree G1."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.rs_view"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
        ]
        self.language_keys = self.language_keys or ["annotation.human.task_description"]
        self.observation_indices = self.observation_indices or _DEFAULT_OBS
        self.action_indices = self.action_indices or _DEFAULT_ACT


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATA_CONFIG_MAP: Dict[str, BaseDataConfig] = {
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "so100_4cam": So100QuadCamDataConfig(),
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "unitree_g1": UnitreeG1DataConfig(),
}


def load_data_config(data_config: Union[str, BaseDataConfig]) -> BaseDataConfig:
    """Load a data configuration by name or pass through an object.

    Args:
        data_config: String name (e.g. ``"so100_dualcam"``) or a ``BaseDataConfig`` instance.

    Returns:
        Resolved ``BaseDataConfig``.

    Raises:
        ValueError: If the name is not found in the registry.
    """
    if isinstance(data_config, BaseDataConfig):
        logger.info(f"Using provided data config: {type(data_config).__name__}")
        return data_config

    if isinstance(data_config, str):
        if data_config in DATA_CONFIG_MAP:
            cfg = DATA_CONFIG_MAP[data_config]
            logger.info(f"Loaded data config '{data_config}': {type(cfg).__name__}")
            return cfg
        raise ValueError(f"Unknown data_config '{data_config}'. Available: {list(DATA_CONFIG_MAP.keys())}")

    raise TypeError(f"data_config must be str or BaseDataConfig, got {type(data_config)}")


def create_custom_data_config(
    name: str,
    video_keys: List[str],
    state_keys: List[str],
    action_keys: List[str],
    language_keys: Optional[List[str]] = None,
    observation_indices: Optional[List[int]] = None,
    action_indices: Optional[List[int]] = None,
) -> BaseDataConfig:
    """Create a custom data configuration at runtime."""

    class _Custom(BaseDataConfig):
        def __init__(self):
            self.video_keys = video_keys
            self.state_keys = state_keys
            self.action_keys = action_keys
            self.language_keys = language_keys or ["annotation.human.task_description"]
            self.observation_indices = observation_indices or [0]
            self.action_indices = action_indices or list(range(16))

    cfg = _Custom()
    logger.info(f"Created custom data config '{name}': video={cfg.video_keys}, state={cfg.state_keys}")
    return cfg


__all__ = [
    "ModalityConfig",
    "BaseDataConfig",
    "So100DataConfig",
    "So100DualCamDataConfig",
    "So100QuadCamDataConfig",
    "FourierGr1ArmsOnlyDataConfig",
    "BimanualPandaGripperDataConfig",
    "UnitreeG1DataConfig",
    "DATA_CONFIG_MAP",
    "load_data_config",
    "create_custom_data_config",
]
