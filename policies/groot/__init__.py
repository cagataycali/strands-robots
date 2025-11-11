#!/usr/bin/env python3
"""
GR00T Policy Implementation with Complete Abstraction

This module provides a completely self-contained GR00T policy implementation
that embeds all necessary Isaac-GR00T components, removing external dependencies.

Key Features:
- Embedded ExternalRobotInferenceClient
- Embedded data configurations
- Support for string names AND data config objects
- No sys.path.append dependencies
- Complete policy abstraction
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np
from .. import Policy
from .client import ExternalRobotInferenceClient
from .data_config import BaseDataConfig, load_data_config

logger = logging.getLogger(__name__)


class Gr00tPolicy(Policy):
    """Complete GR00T policy implementation with embedded components.

    This policy provides a clean interface to GR00T inference services
    while embedding all necessary components to eliminate external dependencies.
    """

    def __init__(self, data_config: Union[str, BaseDataConfig], host: str = "localhost", port: int = 5555, **kwargs):
        """Initialize GR00T policy with embedded components.

        Args:
            data_config: Either a string name (e.g., "so100_dualcam") or a BaseDataConfig object
            host: Inference service host
            port: Inference service port
            **kwargs: Additional GR00T-specific parameters
        """

        # Load data configuration (supports both strings and objects)
        try:
            self.data_config = load_data_config(data_config)
            self.data_config_name = data_config if isinstance(data_config, str) else type(data_config).__name__

            # Initialize embedded GR00T client
            self.policy_client = ExternalRobotInferenceClient(host=host, port=port)

            # Extract modality keys from data config
            self.camera_keys = self.data_config.video_keys
            self.state_keys = self.data_config.state_keys
            self.action_keys = self.data_config.action_keys
            self.language_keys = self.data_config.language_keys
            self.robot_state_keys = []

            logger.info(f"🧠 GR00T Policy: {self.data_config_name}")
            logger.info(f"📹 Camera keys: {self.camera_keys}")
            logger.info(f"🎯 State keys: {self.state_keys}")
            logger.info(f"⚡ Action keys: {self.action_keys}")
            logger.info(f"💬 Language keys: {self.language_keys}")

        except Exception as e:
            # More helpful error message
            if "zmq" in str(e).lower() or "msgpack" in str(e).lower():
                raise ImportError(
                    f"GR00T dependencies not available: {e}. " f"Please install: pip install msgpack pyzmq"
                ) from e
            else:
                raise ImportError(f"GR00T policy initialization failed: {e}") from e

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Set robot state keys from connected robot."""
        self.robot_state_keys = robot_state_keys
        logger.info(f"🔧 GR00T robot state keys: {self.robot_state_keys}")

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from GR00T policy using embedded client.

        Args:
            observation_dict: Robot observations (cameras + state)
            instruction: Natural language instruction
            **kwargs: Additional parameters

        Returns:
            List of action dictionaries for robot execution
        """

        # Build observation dict according to GR00T data config format
        obs_dict = {}

        # Add camera observations
        for video_key in self.camera_keys:
            camera_key = self._map_video_key_to_camera(video_key, observation_dict)
            if camera_key and camera_key in observation_dict:
                obs_dict[video_key] = observation_dict[camera_key]

        # Add state observations
        robot_state = np.array([observation_dict.get(k, 0.0) for k in self.robot_state_keys])
        self._map_robot_state_to_gr00t_state(obs_dict, robot_state)

        # Add language instruction
        if self.language_keys:
            lang_key = self.language_keys[0]
            obs_dict[lang_key] = instruction

        # Add batch dimension for GR00T inference
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # Get action chunk from GR00T policy via embedded client
        action_chunk = self.policy_client.get_action(obs_dict)

        # Convert to robot actions
        return self._convert_to_robot_actions(action_chunk)

    def _map_video_key_to_camera(self, video_key: str, observation_dict: dict) -> str:
        """Map GR00T video key to actual camera key with improved mapping."""

        # Direct mapping (remove video. prefix)
        camera_name = video_key.replace("video.", "")

        if camera_name in observation_dict:
            return camera_name

        # Enhanced mapping with more options
        mapping = {
            "webcam": ["webcam", "front", "wrist", "main"],
            "front": ["front", "webcam", "top", "ego_view", "main"],
            "wrist": ["wrist", "hand", "end_effector", "gripper"],
            "ego_view": ["front", "ego_view", "webcam", "main"],
            "top": ["top", "overhead", "front"],
            "side": ["side", "lateral", "left", "right"],
        }

        for possible_name in mapping.get(camera_name, [camera_name]):
            if possible_name in observation_dict:
                logger.debug(f"📹 Mapped '{video_key}' -> '{possible_name}'")
                return possible_name

        # Fallback to first available camera
        camera_keys = [k for k in observation_dict.keys() if not k.startswith("state")]
        if camera_keys:
            fallback = camera_keys[0]
            logger.warning(f"⚠️ No direct mapping for '{video_key}', using fallback: '{fallback}'")
            return fallback

        logger.warning(f"⚠️ No camera found for '{video_key}'")
        return None

    def _map_robot_state_to_gr00t_state(self, obs_dict: dict, robot_state: np.ndarray):
        """Map robot state to GR00T state format with enhanced robot support."""

        # SO-100/SO-101 mapping
        if "so100" in self.data_config_name.lower():
            if len(robot_state) >= 6:
                obs_dict["state.single_arm"] = robot_state[:5].astype(np.float64)
                obs_dict["state.gripper"] = robot_state[5:6].astype(np.float64)

        # Fourier GR-1 mapping
        elif "fourier_gr1" in self.data_config_name.lower():
            if len(robot_state) >= 14:
                obs_dict["state.left_arm"] = robot_state[:7].astype(np.float64)
                obs_dict["state.right_arm"] = robot_state[7:14].astype(np.float64)

        # Unitree G1 mapping
        elif "unitree_g1" in self.data_config_name.lower():
            if len(robot_state) >= 14:
                obs_dict["state.left_arm"] = robot_state[:7].astype(np.float64)
                obs_dict["state.right_arm"] = robot_state[7:14].astype(np.float64)

        # Bimanual Panda mapping
        elif "bimanual_panda" in self.data_config_name.lower():
            # More complex mapping for bimanual robots
            if len(robot_state) >= 12:
                obs_dict["state.right_arm_eef_pos"] = robot_state[:3].astype(np.float64)
                obs_dict["state.right_arm_eef_quat"] = robot_state[3:7].astype(np.float64)
                obs_dict["state.left_arm_eef_pos"] = robot_state[7:10].astype(np.float64)
                obs_dict["state.left_arm_eef_quat"] = robot_state[10:14].astype(np.float64)

        else:
            # Generic fallback - use first available state key
            if self.state_keys and len(robot_state) > 0:
                obs_dict[self.state_keys[0]] = robot_state.astype(np.float64)
                logger.debug(f"🎯 Generic state mapping: {self.state_keys[0]} -> {len(robot_state)} values")

    def _convert_to_robot_actions(self, action_chunk: dict) -> List[Dict[str, Any]]:
        """Convert GR00T action chunk to robot action list with improved robustness."""

        # Find action horizon from first available action
        first_action_key = None
        for action_key in self.action_keys:
            modality = action_key.split(".")[-1]
            if f"action.{modality}" in action_chunk:
                first_action_key = f"action.{modality}"
                break

        # Fallback: use any available action key
        if not first_action_key:
            action_keys = [k for k in action_chunk.keys() if k.startswith("action.")]
            first_action_key = action_keys[0] if action_keys else None

        if not first_action_key:
            logger.warning("⚠️ No action keys found in action chunk")
            return []

        action_horizon = action_chunk[first_action_key].shape[0]
        logger.debug(f"⚡ Action horizon: {action_horizon}")

        # Convert each action in chunk
        robot_actions = []
        for i in range(action_horizon):
            action_dict = {}

            # Concatenate all action modalities
            action_parts = []
            for action_key in self.action_keys:
                modality = action_key.split(".")[-1]
                if f"action.{modality}" in action_chunk:
                    action_data = action_chunk[f"action.{modality}"][i]
                    action_parts.append(np.atleast_1d(action_data))

            # Fallback: use all available actions if modality mapping fails
            if not action_parts:
                for key, value in action_chunk.items():
                    if key.startswith("action."):
                        action_parts.append(np.atleast_1d(value[i]))

            # Concatenate actions or use zeros as final fallback
            if action_parts:
                concat_action = np.concatenate(action_parts, axis=0)
            else:
                concat_action = np.zeros(len(self.robot_state_keys) or 6)
                logger.warning(f"⚠️ No actions found, using zeros: shape {concat_action.shape}")

            # Map to robot state keys
            for j, key in enumerate(self.robot_state_keys):
                if j < len(concat_action):
                    action_dict[key] = float(concat_action[j])
                else:
                    action_dict[key] = 0.0

            robot_actions.append(action_dict)

        logger.debug(f"⚡ Generated {len(robot_actions)} robot actions")
        return robot_actions


__all__ = ["Gr00tPolicy"]
