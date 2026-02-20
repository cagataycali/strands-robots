#!/usr/bin/env python3
"""GR00T data configurations — robot embodiment key mappings.

SPDX-License-Identifier: Apache-2.0
"""

# Each config: (video_keys, state_keys, action_keys, language_keys)
DATA_CONFIGS = {
    "so100": {
        "video": ["video.webcam"],
        "state": ["state.single_arm", "state.gripper"],
        "action": ["action.single_arm", "action.gripper"],
        "language": ["annotation.human.task_description"],
    },
    "so100_dualcam": {
        "video": ["video.front", "video.wrist"],
        "state": ["state.single_arm", "state.gripper"],
        "action": ["action.single_arm", "action.gripper"],
        "language": ["annotation.human.task_description"],
    },
    "so100_4cam": {
        "video": ["video.front", "video.wrist", "video.top", "video.side"],
        "state": ["state.single_arm", "state.gripper"],
        "action": ["action.single_arm", "action.gripper"],
        "language": ["annotation.human.task_description"],
    },
    "fourier_gr1_arms_only": {
        "video": ["video.ego_view"],
        "state": ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"],
        "action": ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"],
        "language": ["annotation.human.action.task_description"],
    },
    "bimanual_panda_gripper": {
        "video": ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"],
        "state": [
            "state.right_arm_eef_pos",
            "state.right_arm_eef_quat",
            "state.right_gripper_qpos",
            "state.left_arm_eef_pos",
            "state.left_arm_eef_quat",
            "state.left_gripper_qpos",
        ],
        "action": [
            "action.right_arm_eef_pos",
            "action.right_arm_eef_rot",
            "action.right_gripper_close",
            "action.left_arm_eef_pos",
            "action.left_arm_eef_rot",
            "action.left_gripper_close",
        ],
        "language": ["annotation.human.action.task_description"],
    },
    "unitree_g1": {
        "video": ["video.rs_view"],
        "state": ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand"],
        "action": ["action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"],
        "language": ["annotation.human.task_description"],
    },
}


def load_data_config(name):
    """Load a data config by name. Returns dict with video/state/action/language keys."""
    if isinstance(name, dict):
        return name
    if name not in DATA_CONFIGS:
        raise ValueError(f"Unknown data_config '{name}'. Available: {list(DATA_CONFIGS.keys())}")
    return DATA_CONFIGS[name]
