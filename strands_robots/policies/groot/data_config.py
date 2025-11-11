#!/usr/bin/env python3
"""
GR00T Data Configuration Abstraction

Embedded Isaac-GR00T data configurations to eliminate external dependencies.
Extracted and simplified from Isaac-GR00T for complete policy abstraction.

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """Configuration for a modality (cameras, state, actions, language).

    This is a simplified version of the Isaac-GR00T ModalityConfig
    that contains only the essential fields needed for policy abstraction.
    """

    delta_indices: List[int]
    """Delta indices to sample relative to the current index."""

    modality_keys: List[str]
    """The keys to load for the modality in the dataset."""

    def model_dump_json(self) -> str:
        """Serialize to JSON string for compatibility."""
        import json

        return json.dumps({"delta_indices": self.delta_indices, "modality_keys": self.modality_keys})


@dataclass
class BaseDataConfig(ABC):
    """Abstract base class for GR00T data configurations.

    Defines the interface that all data configurations must implement,
    specifying camera keys, state keys, action keys, and language keys.
    """

    # Subclasses must define these
    video_keys: List[str]
    state_keys: List[str]
    action_keys: List[str]
    language_keys: List[str]
    observation_indices: List[int]
    action_indices: List[int]

    def modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration for this data config."""
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }


# ===================================================================
# Concrete Data Configurations
# ===================================================================


@dataclass
class So100DataConfig(BaseDataConfig):
    """SO-100 single camera configuration."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        if self.video_keys is None:
            self.video_keys = ["video.webcam"]
        if self.state_keys is None:
            self.state_keys = ["state.single_arm", "state.gripper"]
        if self.action_keys is None:
            self.action_keys = ["action.single_arm", "action.gripper"]
        if self.language_keys is None:
            self.language_keys = ["annotation.human.task_description"]
        if self.observation_indices is None:
            self.observation_indices = [0]
        if self.action_indices is None:
            self.action_indices = list(range(16))


@dataclass
class So100DualCamDataConfig(BaseDataConfig):
    """SO-100 dual camera configuration."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        if self.video_keys is None:
            self.video_keys = ["video.front", "video.wrist"]
        if self.state_keys is None:
            self.state_keys = ["state.single_arm", "state.gripper"]
        if self.action_keys is None:
            self.action_keys = ["action.single_arm", "action.gripper"]
        if self.language_keys is None:
            self.language_keys = ["annotation.human.task_description"]
        if self.observation_indices is None:
            self.observation_indices = [0]
        if self.action_indices is None:
            self.action_indices = list(range(16))


@dataclass
class So100QuadCamDataConfig(BaseDataConfig):
    """SO-100 quad camera configuration."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        if self.video_keys is None:
            self.video_keys = ["video.front", "video.wrist", "video.top", "video.side"]
        if self.state_keys is None:
            self.state_keys = ["state.single_arm", "state.gripper"]
        if self.action_keys is None:
            self.action_keys = ["action.single_arm", "action.gripper"]
        if self.language_keys is None:
            self.language_keys = ["annotation.human.task_description"]
        if self.observation_indices is None:
            self.observation_indices = [0]
        if self.action_indices is None:
            self.action_indices = list(range(16))


@dataclass
class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    """Fourier GR-1 arms only configuration."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        if self.video_keys is None:
            self.video_keys = ["video.ego_view"]
        if self.state_keys is None:
            self.state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
        if self.action_keys is None:
            self.action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
        if self.language_keys is None:
            self.language_keys = ["annotation.human.action.task_description"]
        if self.observation_indices is None:
            self.observation_indices = [0]
        if self.action_indices is None:
            self.action_indices = list(range(16))


@dataclass
class BimanualPandaGripperDataConfig(BaseDataConfig):
    """Bimanual Panda gripper configuration."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        if self.video_keys is None:
            self.video_keys = ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"]
        if self.state_keys is None:
            self.state_keys = [
                "state.right_arm_eef_pos",
                "state.right_arm_eef_quat",
                "state.right_gripper_qpos",
                "state.left_arm_eef_pos",
                "state.left_arm_eef_quat",
                "state.left_gripper_qpos",
            ]
        if self.action_keys is None:
            self.action_keys = [
                "action.right_arm_eef_pos",
                "action.right_arm_eef_rot",
                "action.right_gripper_close",
                "action.left_arm_eef_pos",
                "action.left_arm_eef_rot",
                "action.left_gripper_close",
            ]
        if self.language_keys is None:
            self.language_keys = ["annotation.human.action.task_description"]
        if self.observation_indices is None:
            self.observation_indices = [0]
        if self.action_indices is None:
            self.action_indices = list(range(16))


@dataclass
class UnitreeG1DataConfig(BaseDataConfig):
    """Unitree G1 configuration."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        if self.video_keys is None:
            self.video_keys = ["video.rs_view"]
        if self.state_keys is None:
            self.state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"]
        if self.action_keys is None:
            self.action_keys = ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"]
        if self.language_keys is None:
            self.language_keys = ["annotation.human.task_description"]
        if self.observation_indices is None:
            self.observation_indices = [0]
        if self.action_indices is None:
            self.action_indices = list(range(16))


# ===================================================================
# Data Configuration Registry
# ===================================================================

# Global registry of available data configurations
DATA_CONFIG_MAP: Dict[str, BaseDataConfig] = {
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "so100_4cam": So100QuadCamDataConfig(),
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "unitree_g1": UnitreeG1DataConfig(),
}


def load_data_config(data_config: Union[str, BaseDataConfig]) -> BaseDataConfig:
    """Load a data configuration from string name or return the object directly.

    Args:
        data_config: Either a string name (e.g., "so100_dualcam") or a BaseDataConfig object

    Returns:
        BaseDataConfig instance

    Raises:
        ValueError: If string name is not found in registry
    """

    # If it's already a data config object, return it directly
    if isinstance(data_config, BaseDataConfig):
        logger.info(f"✅ Using provided data config object: {type(data_config).__name__}")
        return data_config

    # If it's a string, look it up in the registry
    elif isinstance(data_config, str):
        if data_config in DATA_CONFIG_MAP:
            config = DATA_CONFIG_MAP[data_config]
            logger.info(f"✅ Loaded data config '{data_config}': {type(config).__name__}")
            return config
        else:
            available = list(DATA_CONFIG_MAP.keys())
            raise ValueError(f"❌ Invalid data_config '{data_config}'. " f"Available options: {available}")

    else:
        raise ValueError(f"❌ data_config must be str or BaseDataConfig, got {type(data_config)}")


def create_custom_data_config(
    name: str,
    video_keys: List[str],
    state_keys: List[str],
    action_keys: List[str],
    language_keys: Optional[List[str]] = None,
    observation_indices: Optional[List[int]] = None,
    action_indices: Optional[List[int]] = None,
) -> BaseDataConfig:
    """Create a custom data configuration.

    Args:
        name: Name for the configuration (for logging)
        video_keys: List of camera/video observation keys
        state_keys: List of robot state keys
        action_keys: List of robot action keys
        language_keys: List of language instruction keys
        observation_indices: Observation delta indices
        action_indices: Action delta indices

    Returns:
        Custom BaseDataConfig instance
    """

    class CustomDataConfig(BaseDataConfig):
        def __init__(self):
            self.video_keys = video_keys
            self.state_keys = state_keys
            self.action_keys = action_keys
            self.language_keys = language_keys or ["annotation.human.task_description"]
            self.observation_indices = observation_indices or [0]
            self.action_indices = action_indices or list(range(16))

    config = CustomDataConfig()

    logger.info(f"✅ Created custom data config '{name}':")
    logger.info(f"   📹 Video keys: {config.video_keys}")
    logger.info(f"   🎯 State keys: {config.state_keys}")
    logger.info(f"   ⚡ Action keys: {config.action_keys}")

    return config


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
