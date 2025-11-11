#!/usr/bin/env python3
"""
Universal Robot Control with Policy Abstraction for Any VLA Provider

This module provides a clean robot interface that works with any LeRobot-compatible
robot and any VLA provider through the Policy abstraction.
"""

import asyncio
import logging
import time
from typing import Any, Dict, AsyncGenerator, Optional, Union, List

import numpy as np
from lerobot.robots.robot import Robot as LeRobotRobot
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from strands.types.tools import ToolUse, ToolResult, ToolSpec
from strands.types._events import ToolResultEvent
from strands.tools.tools import AgentTool

from .policies import Policy, create_policy

logger = logging.getLogger(__name__)


class Robot(AgentTool):
    """Universal robot control with policy abstraction for any LeRobot-compatible robot."""

    def __init__(
        self,
        tool_name: str,
        robot: Union[LeRobotRobot, RobotConfig, str],
        cameras: Optional[Dict[str, Dict[str, Any]]] = None,
        action_horizon: int = 8,
        data_config: Union[str, Any, None] = None,
        **kwargs,
    ):
        """Initialize Robot - minimal, policy specified at invocation time.

        Args:
            tool_name: Name for this robot tool
            robot: LeRobot Robot instance, RobotConfig, or robot type string
            cameras: Camera configuration dict:
                {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}
            action_horizon: Actions per inference step
            data_config: Data configuration (for GR00T compatibility)
            **kwargs: Robot-specific parameters (port, etc.)
        """
        super().__init__()

        self.tool_name_str = tool_name
        self.action_horizon = action_horizon
        self.data_config = data_config

        # Initialize robot using lerobot's abstraction
        self.robot = self._initialize_robot(robot, cameras, **kwargs)

        logger.info(f"🤖 {tool_name} initialized")
        logger.info(f"📱 Robot: {self.robot.name} (type: {getattr(self.robot, 'robot_type', 'unknown')})")

        # Get camera info if available
        if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
            cameras_list = list(self.robot.config.cameras.keys())
            logger.info(f"📹 Cameras: {cameras_list}")

        if data_config:
            logger.info(f"⚙️ Data config: {data_config}")

    def _initialize_robot(
        self, robot: Union[LeRobotRobot, RobotConfig, str], cameras: Optional[Dict[str, Dict[str, Any]]], **kwargs
    ) -> LeRobotRobot:
        """Initialize LeRobot robot instance using native lerobot patterns."""

        # Direct robot instance - use as-is
        if isinstance(robot, LeRobotRobot):
            return robot

        # Robot config - use lerobot's factory
        elif isinstance(robot, RobotConfig):
            return make_robot_from_config(robot)

        # Robot type string - create config and use lerobot's factory
        elif isinstance(robot, str):
            config = self._create_minimal_config(robot, cameras, **kwargs)
            return make_robot_from_config(config)

        else:
            raise ValueError(
                f"Unsupported robot type: {type(robot)}. "
                f"Expected LeRobot Robot instance, RobotConfig, or robot type string."
            )

    def _create_minimal_config(
        self, robot_type: str, cameras: Optional[Dict[str, Dict[str, Any]]], **kwargs
    ) -> RobotConfig:
        """Create minimal robot config using specific robot config classes."""

        # Convert cameras to lerobot format
        camera_configs = {}
        if cameras:
            for name, config in cameras.items():
                if config.get("type", "opencv") == "opencv":
                    camera_configs[name] = OpenCVCameraConfig(
                        index_or_path=config["index_or_path"],
                        fps=config.get("fps", 30),
                        width=config.get("width", 640),
                        height=config.get("height", 480),
                        rotation=config.get("rotation", 0),
                        color_mode=config.get("color_mode", "rgb"),
                    )
                else:
                    raise ValueError(f"Unsupported camera type: {config.get('type')}")

        # Map robot type to specific config class
        config_mapping = {
            "so101_follower": ("lerobot.robots.so101_follower", "SO101FollowerConfig"),
            "so100_follower": ("lerobot.robots.so100_follower", "SO100FollowerConfig"),
            "bi_so100_follower": ("lerobot.robots.bi_so100_follower", "BiSO100FollowerConfig"),
            "viperx": ("lerobot.robots.viperx", "ViperXConfig"),
            "koch_follower": ("lerobot.robots.koch_follower", "KochFollowerConfig"),
            # Add more as needed
        }

        if robot_type not in config_mapping:
            raise ValueError(
                f"Unsupported robot type: {robot_type}. " f"Supported types: {list(config_mapping.keys())}"
            )

        # Import specific config class dynamically
        module_name, class_name = config_mapping[robot_type]
        try:
            import importlib

            module = importlib.import_module(module_name)
            ConfigClass = getattr(module, class_name)
        except Exception as e:
            raise ValueError(f"Failed to import {class_name} from {module_name}: {e}")

        # Create config with proper parameters
        config_data = {
            "id": self.tool_name_str,
            "cameras": camera_configs,
        }

        # Filter kwargs to only include supported fields for this robot type
        # Port is common for most serial robots
        if "port" in kwargs:
            config_data["port"] = kwargs["port"]

        # Add other common fields as needed
        for key in ["calibration_dir", "mock", "use_degrees"]:
            if key in kwargs:
                config_data[key] = kwargs[key]

        try:
            return ConfigClass(**config_data)
        except Exception as e:
            raise ValueError(
                f"Failed to create {class_name} for robot type '{robot_type}': {e}. " f"Config: {config_data}"
            )

    async def _get_policy(
        self, policy_port: Optional[int] = None, policy_host: str = "localhost", policy_provider: str = "groot"
    ) -> Policy:
        """Create policy on-the-fly from invocation parameters."""

        if not policy_port:
            raise ValueError("policy_port is required for robot operation")

        policy_config = {"port": policy_port, "host": policy_host}

        if self.data_config:
            policy_config["data_config"] = self.data_config

        return create_policy(policy_provider, **policy_config)

    async def _connect_robot(self) -> bool:
        """Connect to robot hardware."""
        if self.robot.is_connected:
            return True

        try:
            logger.info(f"🔌 Connecting to {self.robot}...")
            await asyncio.to_thread(self.robot.connect)

            if not self.robot.is_connected:
                logger.error(f"❌ Failed to connect to {self.robot}")
                return False

            logger.info(f"✅ {self.robot} connected")
            return True

        except Exception as e:
            logger.error(f"❌ Robot connection failed: {e}")
            return False

    async def _initialize_policy(self, policy: Policy) -> bool:
        """Initialize policy with robot state keys."""
        try:
            # Get robot state keys from observation
            test_obs = await asyncio.to_thread(self.robot.get_observation)

            # Filter out camera keys to get robot state keys
            camera_keys = []
            if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras"):
                camera_keys = list(self.robot.config.cameras.keys())

            robot_state_keys = [k for k in test_obs.keys() if k not in camera_keys]

            # Set robot state keys in policy
            policy.set_robot_state_keys(robot_state_keys)
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize policy: {e}")
            return False

    async def _execute_task(
        self,
        instruction: str,
        policy_port: Optional[int] = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
    ) -> Dict[str, Any]:
        """Execute robot task using specified policy."""

        # Connect to robot
        if not await self._connect_robot():
            return {
                "status": "error",
                "content": [{"text": f"❌ Failed to connect to {self.tool_name_str}"}],
            }

        try:
            # Get policy instance
            policy_instance = await self._get_policy(policy_port, policy_host, policy_provider)

            # Initialize policy with robot state keys
            if not await self._initialize_policy(policy_instance):
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Failed to initialize policy"}],
                }

            logger.info(f"🎯 Executing: '{instruction}' on {self.tool_name_str}")
            logger.info(f"🧠 Using policy: {policy_provider} on {policy_host}:{policy_port}")

            start_time = time.time()
            step_count = 0

            while time.time() - start_time < duration:
                # Get observation from robot
                observation = await asyncio.to_thread(self.robot.get_observation)

                # Get actions from policy
                robot_actions = await policy_instance.get_actions(observation, instruction)

                # Execute actions from chunk
                for action_dict in robot_actions[: self.action_horizon]:
                    await asyncio.to_thread(self.robot.send_action, action_dict)
                    step_count += 1

                await asyncio.sleep(0.01)

            elapsed = time.time() - start_time

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"✅ Task completed: '{instruction}'\n"
                        f"🤖 Robot: {self.tool_name_str} ({self.robot})\n"
                        f"🧠 Policy: {policy_provider} on {policy_host}:{policy_port}\n"
                        f"⏱️ Duration: {elapsed:.1f}s\n"
                        f"🎯 Steps: {step_count}"
                    }
                ],
            }

        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"❌ {self.tool_name_str} task failed: {str(e)}"}],
            }

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "robot"

    @property
    def tool_spec(self) -> ToolSpec:
        """Get tool specification."""
        return {
            "name": self.tool_name_str,
            "description": f"Universal robot control ({self.robot}). Policy specified at invocation time.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction (e.g., 'Pick up the red block', 'Wave hello')",
                        },
                        "policy_port": {
                            "type": "integer",
                            "description": "Policy service port (e.g., 8000 for GR00T)",
                        },
                        "policy_host": {
                            "type": "string",
                            "description": "Policy service host (default: localhost)",
                            "default": "localhost",
                        },
                        "policy_provider": {
                            "type": "string",
                            "description": "Policy provider (groot, openai, etc.)",
                            "default": "groot",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Maximum execution time in seconds",
                            "default": 30.0,
                        },
                    },
                    "required": ["instruction", "policy_port"],
                }
            },
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Stream robot task execution."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            instruction = input_data.get("instruction", "")
            policy_port = input_data.get("policy_port")
            policy_host = input_data.get("policy_host", "localhost")
            policy_provider = input_data.get("policy_provider", "groot")
            duration = input_data.get("duration", 30.0)

            if not instruction:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"❌ No instruction provided"}],
                    }
                )
                return

            if not policy_port:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"❌ policy_port is required"}],
                    }
                )
                return

            # Execute task
            task_result = await self._execute_task(instruction, policy_port, policy_host, policy_provider, duration)
            result = {"toolUseId": tool_use_id, **task_result}
            yield ToolResultEvent(result)

        except Exception as e:
            logger.error(f"❌ {self.tool_name_str} error: {e}")
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use.get("toolUseId", ""),
                    "status": "error",
                    "content": [{"text": f"❌ {self.tool_name_str} error: {str(e)}"}],
                }
            )

    async def get_status(self) -> Dict[str, Any]:
        """Get robot status."""
        return {
            "robot_name": self.tool_name_str,
            "robot_type": getattr(self.robot, "robot_type", self.robot.name),
            "robot_info": str(self.robot),
            "data_config": self.data_config,
            "is_connected": self.robot.is_connected,
            "cameras": (
                list(self.robot.config.cameras.keys())
                if hasattr(self.robot, "config") and hasattr(self.robot.config, "cameras")
                else []
            ),
        }

    async def stop(self):
        """Stop robot and disconnect."""
        try:
            if hasattr(self.robot, "disconnect"):
                await asyncio.to_thread(self.robot.disconnect)
            logger.info(f"🛑 {self.tool_name_str} stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping robot: {e}")
