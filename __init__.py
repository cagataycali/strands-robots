#!/usr/bin/env python3
"""
Strands Robotics - Universal Robot Control with Policy Abstraction

A unified Python interface for controlling diverse robot hardware through
any VLA provider with clean policy abstraction architecture.

Key features:
- Policy abstraction for any VLA provider (GR00T, ACT, SmolVLA, etc.)
- Universal robot support through LeRobot integration
- Clean separation between robot control and policy inference
- Direct policy injection for maximum flexibility
- Multi-camera support with rich configuration options
"""

import warnings

try:
    from .robot import Robot
    from .policies import Policy, MockPolicy, create_policy

    # Import tools
    from .tools.gr00t_inference import gr00t_inference
    from .tools.lerobot_camera import lerobot_camera
    from .tools.serial_tool import serial_tool

    try:
        from .policies.groot import Gr00tPolicy

        __all__ = [
            "Robot",
            "Policy",
            "Gr00tPolicy",
            "MockPolicy",
            "create_policy",
            "gr00t_inference",
            "lerobot_camera",
            "serial_tool",
        ]
    except ImportError as e:
        warnings.warn(f"GR00T policy not available (missing dependencies): {e}")
        __all__ = ["Robot", "Policy", "MockPolicy", "create_policy", "gr00t_inference", "lerobot_camera", "serial_tool"]

except ImportError as e:
    warnings.warn(f"Could not import core components: {e}")
    __all__ = []

__version__ = "0.2.0"
