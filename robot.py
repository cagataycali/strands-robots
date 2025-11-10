#!/usr/bin/env python3
"""
Universal Robot Control with Policy Abstraction for Any VLA Provider

This module provides a clean robot interface that can work with any VLA provider
through the Policy abstraction, removing hardcoded dependencies on specific models.
"""

import asyncio
import logging
import time
from typing import Any, Dict, AsyncGenerator, Optional, Union, List

import numpy as np
from lerobot.robots.robot import Robot as LeRobotRobot
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from strands.types.tools import ToolUse, ToolResult, ToolSpec
from strands.types._events import ToolResultEvent
from strands.tools.tools import AgentTool

from .policies import Policy, create_policy

logger = logging.getLogger(__name__)


class Robot(AgentTool):
    """Universal robot control with policy abstraction for any VLA provider."""

    def __init__(
        self,
        tool_name: str,
        robot: Union[LeRobotRobot, RobotConfig, str],
        cameras: Optional[Dict[str, Union[str, Dict[str, Any]]]] = None,
        action_horizon: int = 8,
        data_config: Union[str, Any, None] = None,
        default_policy: Optional[Union[Policy, str, Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Initialize Robot with policy abstraction.
        
        Args:
            tool_name: Name for this robot tool
            robot: Robot instance, config, or type string
            cameras: Camera configuration dict with structured format:
                {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}
            action_horizon: Actions per inference step
            data_config: Data configuration (for GR00T compatibility)
            default_policy: Default policy to use (Policy instance, provider string, or config dict)
            **kwargs: Robot-specific parameters (port, etc.)
        """
        super().__init__()
        
        self.tool_name_str = tool_name
        self.action_horizon = action_horizon
        self.data_config = data_config
        self.robot_kwargs = kwargs
        
        # Initialize robot
        self.robot = self._initialize_robot(robot, cameras, **kwargs)
        self._robot_initialized = False
        
        # Policy management 
        self.registered_policies: Dict[str, Policy] = {}
        self.default_policy_name: Optional[str] = None
        
        # Register default policy if provided
        if default_policy:
            self._register_default_policy(default_policy)
        
        logger.info(f"🤖 {tool_name} initialized")
        logger.info(f"📱 Robot: {self.robot.name}")
        logger.info(f"📹 Cameras: {list(self.robot.config.cameras.keys()) if hasattr(self.robot.config, 'cameras') else []}")
        if data_config:
            logger.info(f"⚙️ Data config: {data_config}")

    def _initialize_robot(
        self, 
        robot: Union[LeRobotRobot, RobotConfig, str], 
        cameras: Optional[Dict[str, Union[str, Dict[str, Any]]]], 
        **kwargs
    ) -> LeRobotRobot:
        """Initialize LeRobot robot instance."""
        
        if isinstance(robot, LeRobotRobot):
            return robot
        elif isinstance(robot, RobotConfig):
            return make_robot_from_config(robot)
        elif isinstance(robot, str):
            robot_config = self._create_robot_config(robot, cameras, **kwargs)
            return make_robot_from_config(robot_config)
        else:
            raise ValueError(f"Unsupported robot type: {type(robot)}")

    def _create_robot_config(
        self, 
        robot_type: str, 
        cameras: Optional[Dict[str, Union[str, Dict[str, Any]]]], 
        **kwargs
    ) -> RobotConfig:
        """Create robot config from type string with new camera format support."""
        
        # Convert cameras to CameraConfig objects (structured format only)
        camera_configs = {}
        if cameras:
            for name, config in cameras.items():
                if isinstance(config, dict):
                    # Structured format: {"wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30}}
                    if config.get("type", "opencv") == "opencv":
                        camera_configs[name] = OpenCVCameraConfig(
                            index_or_path=config["index_or_path"],
                            fps=config.get("fps", 30),
                            width=config.get("width", 640), 
                            height=config.get("height", 480),
                            rotation=config.get("rotation", 0),
                            color_mode=config.get("color_mode", "rgb")
                        )
                else:
                    raise ValueError(f"Camera '{name}' must use structured format: "
                                   f"{{'type': 'opencv', 'index_or_path': '/dev/video0', 'fps': 30}}")   
        
        # Create robot config based on type  
        config_map = {
            "so100_follower": ("lerobot.robots.so100_follower", "SO100FollowerConfig"),
            "so101_follower": ("lerobot.robots.so101_follower", "SO101FollowerConfig"),
            "lekiwi": ("lerobot.robots.lekiwi", "LeKiwiConfig"),
            "viperx": ("lerobot.robots.viperx", "ViperXConfig"),
            "koch_follower": ("lerobot.robots.koch_follower", "KochFollowerConfig"),
            "hope_jr_arm": ("lerobot.robots.hope_jr", "HopeJrArmConfig"),
            "hope_jr_hand": ("lerobot.robots.hope_jr", "HopeJrHandConfig"),
            "bi_so100_follower": ("lerobot.robots.bi_so100_follower", "BiSO100FollowerConfig"),
            "reachy2": ("lerobot.robots.reachy2", "Reachy2RobotConfig"),
        }
        
        if robot_type not in config_map:
            available = ", ".join(config_map.keys())
            raise ValueError(f"Unknown robot type: {robot_type}. Available: {available}")
            
        module_name, class_name = config_map[robot_type]
        module = __import__(module_name, fromlist=[class_name])
        ConfigClass = getattr(module, class_name)
        
        # Create config with common parameters
        config_kwargs = {
            "cameras": camera_configs,
            "id": kwargs.get("id", self.tool_name_str),
        }
        
        # Add robot-specific parameters
        if robot_type in ["so100_follower", "so101_follower", "lekiwi", "koch_follower", "hope_jr_arm"]:
            config_kwargs["port"] = kwargs.get("port", "/dev/ttyACM0")
        elif robot_type == "hope_jr_hand":
            config_kwargs["port"] = kwargs.get("port", "/dev/ttyACM0")
            config_kwargs["side"] = kwargs.get("side", "right")
        elif robot_type == "bi_so100_follower":
            config_kwargs["left_arm_port"] = kwargs.get("left_arm_port", "/dev/ttyACM0")
            config_kwargs["right_arm_port"] = kwargs.get("right_arm_port", "/dev/ttyACM1")
        elif robot_type == "reachy2":
            config_kwargs["ip_address"] = kwargs.get("ip_address", "localhost")
        
        # Add any remaining kwargs
        config_kwargs.update({k: v for k, v in kwargs.items() if k not in config_kwargs})
        
        return ConfigClass(**config_kwargs)

    def _register_default_policy(self, default_policy: Union[Policy, str, Dict[str, Any]]):
        """Register default policy from various input formats."""
        if isinstance(default_policy, Policy):
            # Direct policy instance
            policy = default_policy
            policy_name = f"default_{policy.provider_name}"
        elif isinstance(default_policy, str):
            # Provider name string
            policy_kwargs = {}
            if self.data_config:
                policy_kwargs["data_config"] = self.data_config
            policy = create_policy(default_policy, **policy_kwargs)
            policy_name = f"default_{default_policy}"
        elif isinstance(default_policy, dict):
            # Policy configuration dict
            provider = default_policy.pop("provider")
            if "data_config" not in default_policy and self.data_config:
                default_policy["data_config"] = self.data_config
            policy = create_policy(provider, **default_policy)
            policy_name = f"default_{provider}"
        else:
            raise ValueError(f"Invalid default_policy type: {type(default_policy)}")
            
        self.register_policy(policy_name, policy)
        self.default_policy_name = policy_name
        logger.info(f"🎯 Registered default policy: {policy_name}")

    def register_policy(self, policy_name: str, policy: Policy):
        """Register a policy for use with this robot.
        
        Args:
            policy_name: Name to identify the policy
            policy: Policy instance
        """
        self.registered_policies[policy_name] = policy
        logger.info(f"📡 Registered policy: {policy_name} ({policy.provider_name})")

    def register_policy_service(self, policy_name: str, port: int, provider: str = "groot", **kwargs):
        """Register a policy service (for backward compatibility).
        
        Args:
            policy_name: Name to identify the policy
            port: Service port
            provider: Policy provider name
            **kwargs: Provider-specific parameters
        """
        policy_config = {"port": port, **kwargs}
        if self.data_config and "data_config" not in policy_config:
            policy_config["data_config"] = self.data_config
            
        policy = create_policy(provider, **policy_config)
        self.register_policy(policy_name, policy)

    async def _get_policy(self, policy_identifier: Union[str, Policy, None]) -> Policy:
        """Get policy from various input formats.
        
        Args:
            policy_identifier: Policy name, Policy instance, or None for default
            
        Returns:
            Policy instance
            
        Raises:
            ValueError: If policy cannot be resolved
        """
        # Direct policy instance
        if isinstance(policy_identifier, Policy):
            return policy_identifier
            
        # Policy name string  
        elif isinstance(policy_identifier, str):
            if policy_identifier in self.registered_policies:
                return self.registered_policies[policy_identifier]
            else:
                # Try as port number for GR00T (backward compatibility)
                try:
                    port = int(policy_identifier)
                    policy_config = {"port": port}
                    if self.data_config:
                        policy_config["data_config"] = self.data_config
                    return create_policy("groot", **policy_config)
                except ValueError:
                    pass
                    
                available = list(self.registered_policies.keys())
                raise ValueError(f"Unknown policy: {policy_identifier}. Available: {available}")
        
        # None - use default
        elif policy_identifier is None:
            if self.default_policy_name and self.default_policy_name in self.registered_policies:
                return self.registered_policies[self.default_policy_name]
            else:
                raise ValueError("No policy specified and no default policy available")
        
        else:
            raise ValueError(f"Invalid policy identifier type: {type(policy_identifier)}")

    async def _connect_robot(self) -> bool:
        """Connect to robot hardware."""
        if self._robot_initialized:
            return True
            
        try:
            logger.info(f"🔌 Connecting to {self.robot.name}...")
            await asyncio.to_thread(self.robot.connect)
            
            if not self.robot.is_connected:
                logger.error(f"❌ Failed to connect to {self.robot.name}")
                return False
                
            logger.info(f"✅ {self.robot.name} connected")
            self._robot_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Robot connection failed: {e}")
            return False

    async def _initialize_policy(self, policy: Policy) -> bool:
        """Initialize policy with robot state keys."""
        try:
            # Get robot state keys
            if hasattr(self.robot, '_motors_ft'):
                robot_state_keys = list(self.robot._motors_ft.keys())
            elif hasattr(self.robot, '_motor_names'):
                robot_state_keys = list(self.robot._motor_names)
            else:
                # Get from observation
                test_obs = await asyncio.to_thread(self.robot.get_observation)
                camera_keys = list(self.robot.config.cameras.keys()) if hasattr(self.robot.config, 'cameras') else []
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
        policy: Union[str, Policy, None] = None,
        duration: float = 30.0
    ) -> Dict[str, Any]:
        """Execute robot task using specified policy.
        
        Args:
            instruction: Natural language instruction
            policy: Policy identifier (name, instance, or None for default)
            duration: Maximum execution time
            
        Returns:
            Task execution result
        """
        
        # Connect to robot
        if not await self._connect_robot():
            return {
                "status": "error",
                "content": [{"text": f"❌ Failed to connect to {self.tool_name_str}"}],
            }

        try:
            # Get policy instance
            policy_instance = await self._get_policy(policy)
            
            # Initialize policy with robot state keys
            if not await self._initialize_policy(policy_instance):
                return {
                    "status": "error",
                    "content": [{"text": f"❌ Failed to initialize policy"}],
                }
            
            logger.info(f"🎯 Executing: '{instruction}' on {self.tool_name_str}")
            logger.info(f"🧠 Using policy: {policy_instance.provider_name}")
            
            start_time = time.time()
            step_count = 0
            
            while time.time() - start_time < duration:
                # Get observation from robot
                observation = await asyncio.to_thread(self.robot.get_observation)
                
                # Get actions from policy
                robot_actions = await policy_instance.get_actions(observation, instruction)
                
                # Execute actions from chunk
                for action_dict in robot_actions[:self.action_horizon]:
                    await asyncio.to_thread(self.robot.send_action, action_dict)
                    step_count += 1
                
                await asyncio.sleep(0.01)
                
            elapsed = time.time() - start_time
            
            return {
                "status": "success", 
                "content": [
                    {
                        "text": f"✅ Task completed: '{instruction}'\n"
                               f"🤖 Robot: {self.tool_name_str} ({self.robot.name})\n" 
                               f"🧠 Policy: {policy_instance.provider_name}\n"
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
        available_policies = list(self.registered_policies.keys())
        policy_desc = f"Available policies: {available_policies}" if available_policies else "Register policies first"
        
        return {
            "name": self.tool_name_str,
            "description": f"Universal robot control with policy abstraction (Robot: {self.robot.name}). {policy_desc}",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string", 
                            "description": "Natural language instruction (e.g., 'Pick up the red block', 'Wave hello')",
                        },
                        "policy": {
                            "type": "string",
                            "description": f"Policy to use. {policy_desc}. Optional if default policy is set.",
                        },
                        "duration": {
                            "type": "number",
                            "description": "Maximum execution time in seconds", 
                            "default": 30.0,
                        },
                    },
                    "required": ["instruction"],
                }
            },
        }

    async def stream(
        self, 
        tool_use: ToolUse, 
        invocation_state: dict[str, Any], 
        **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Stream robot task execution."""
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            instruction = input_data.get("instruction", "")
            policy = input_data.get("policy")  # Optional
            duration = input_data.get("duration", 30.0)

            if not instruction:
                yield ToolResultEvent({
                    "toolUseId": tool_use_id,
                    "status": "error", 
                    "content": [{"text": f"❌ No instruction provided"}],
                })
                return

            # Execute task (policy is optional if default is set)
            task_result = await self._execute_task(instruction, policy, duration)
            result = {"toolUseId": tool_use_id, **task_result}
            yield ToolResultEvent(result)

        except Exception as e:
            logger.error(f"❌ {self.tool_name_str} error: {e}")
            yield ToolResultEvent({
                "toolUseId": tool_use.get("toolUseId", ""),
                "status": "error",
                "content": [{"text": f"❌ {self.tool_name_str} error: {str(e)}"}],
            })

    async def get_status(self) -> Dict[str, Any]:
        """Get robot status."""
        return {
            "robot_name": self.tool_name_str,
            "robot_type": self.robot.name,
            "data_config": self.data_config,
            "registered_policies": {name: policy.provider_name for name, policy in self.registered_policies.items()},
            "default_policy": self.default_policy_name,
            "is_connected": getattr(self.robot, 'is_connected', False),
            "cameras": list(self.robot.config.cameras.keys()) if hasattr(self.robot.config, 'cameras') else [],
            "initialized": self._robot_initialized,
        }

    async def stop(self):
        """Stop robot and disconnect."""
        try:
            if hasattr(self.robot, 'disconnect'):
                await asyncio.to_thread(self.robot.disconnect)
            self._robot_initialized = False
            logger.info(f"🛑 {self.tool_name_str} stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping robot: {e}")