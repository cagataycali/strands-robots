#!/usr/bin/env python3
"""GR00T Policy — natural language robot control via GR00T inference servers.

SPDX-License-Identifier: Apache-2.0
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np

from .. import Policy
from .client import GR00TClient
from .data_config import load_data_config

logger = logging.getLogger(__name__)


class Gr00tPolicy(Policy):
    """GR00T policy: connects to a GR00T inference server via ZMQ."""

    def __init__(self, data_config: Union[str, dict], host: str = "localhost", port: int = 5555, **kwargs):
        """Initialize GR00T policy.

        Args:
            data_config: Config name (e.g. "so100_dualcam") or dict with video/state/action/language keys
            host: Inference service host
            port: Inference service port
        """
        self.config = load_data_config(data_config)
        self.data_config_name = data_config if isinstance(data_config, str) else "custom"
        self.client = GR00TClient(host=host, port=port)

        self.camera_keys = self.config["video"]
        self.state_keys = self.config["state"]
        self.action_keys = self.config["action"]
        self.language_keys = self.config["language"]
        self.robot_state_keys = []

        logger.info(f"🧠 GR00T Policy: {self.data_config_name} @ {host}:{port}")

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from GR00T policy server.

        Args:
            observation_dict: Robot observations (cameras + state)
            instruction: Natural language instruction

        Returns:
            List of action dicts for robot execution
        """
        obs = {}

        # Camera observations
        for vkey in self.camera_keys:
            cam = self._find_camera(vkey, observation_dict)
            if cam and cam in observation_dict:
                obs[vkey] = observation_dict[cam]

        # State observations
        robot_state = np.array([observation_dict.get(k, 0.0) for k in self.robot_state_keys])
        self._map_state(obs, robot_state)

        # Language instruction
        if self.language_keys:
            obs[self.language_keys[0]] = instruction

        # Batch dimension
        for k in obs:
            if isinstance(obs[k], np.ndarray):
                obs[k] = obs[k][np.newaxis, ...]
            else:
                obs[k] = [obs[k]]

        action_chunk = self.client.get_action(obs)
        return self._to_robot_actions(action_chunk)

    def _find_camera(self, video_key: str, obs: dict) -> str:
        """Map GR00T video key to available camera key."""
        name = video_key.replace("video.", "")
        if name in obs:
            return name

        aliases = {
            "webcam": ["webcam", "front", "wrist", "main"],
            "front": ["front", "webcam", "top", "ego_view", "main"],
            "wrist": ["wrist", "hand", "end_effector", "gripper"],
            "ego_view": ["front", "ego_view", "webcam", "main"],
            "top": ["top", "overhead", "front"],
            "side": ["side", "lateral", "left", "right"],
            "rs_view": ["rs_view", "front", "ego_view", "webcam"],
        }
        for candidate in aliases.get(name, [name]):
            if candidate in obs:
                return candidate

        # Fallback: first non-state key
        cams = [k for k in obs if not k.startswith("state")]
        return cams[0] if cams else None

    def _map_state(self, obs: dict, state: np.ndarray):
        """Map robot state array to GR00T state keys."""
        name = self.data_config_name.lower()
        if "so100" in name and len(state) >= 6:
            obs["state.single_arm"] = state[:5].astype(np.float64)
            obs["state.gripper"] = state[5:6].astype(np.float64)
        elif "fourier_gr1" in name and len(state) >= 14:
            obs["state.left_arm"] = state[:7].astype(np.float64)
            obs["state.right_arm"] = state[7:14].astype(np.float64)
        elif "unitree_g1" in name and len(state) >= 14:
            obs["state.left_arm"] = state[:7].astype(np.float64)
            obs["state.right_arm"] = state[7:14].astype(np.float64)
        elif "bimanual_panda" in name and len(state) >= 12:
            obs["state.right_arm_eef_pos"] = state[:3].astype(np.float64)
            obs["state.right_arm_eef_quat"] = state[3:7].astype(np.float64)
            obs["state.left_arm_eef_pos"] = state[7:10].astype(np.float64)
            obs["state.left_arm_eef_quat"] = state[10:14].astype(np.float64)
        elif self.state_keys and len(state) > 0:
            obs[self.state_keys[0]] = state.astype(np.float64)

    def _to_robot_actions(self, chunk: dict) -> List[Dict[str, Any]]:
        """Convert GR00T action chunk to list of robot action dicts."""
        # Find first action key to get horizon
        act_key = None
        for k in self.action_keys:
            mod = k.split(".")[-1]
            if f"action.{mod}" in chunk:
                act_key = f"action.{mod}"
                break
        if not act_key:
            act_keys = [k for k in chunk if k.startswith("action.")]
            act_key = act_keys[0] if act_keys else None
        if not act_key:
            return []

        horizon = chunk[act_key].shape[0]
        actions = []
        for i in range(horizon):
            parts = []
            for k in self.action_keys:
                mod = k.split(".")[-1]
                if f"action.{mod}" in chunk:
                    parts.append(np.atleast_1d(chunk[f"action.{mod}"][i]))
            if not parts:
                for k, v in chunk.items():
                    if k.startswith("action."):
                        parts.append(np.atleast_1d(v[i]))

            concat = np.concatenate(parts) if parts else np.zeros(len(self.robot_state_keys) or 6)
            actions.append(
                {k: float(concat[j]) if j < len(concat) else 0.0 for j, k in enumerate(self.robot_state_keys)}
            )

        return actions


__all__ = ["Gr00tPolicy"]
