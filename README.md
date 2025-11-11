# strands-robots

Robot control for Strands Agents.

## Install

```bash
pip install strands-robots
```

## Quick Start

```python
from strands import Agent
from strands_robots import Robot, gr00t_inference

# Create robot with cameras
robot = Robot(
    tool_name="my_arm",
    robot="so101_follower",
    cameras={
        "front": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30}
    },
    port="/dev/ttyACM0",
    data_config="so100_dualcam"
)

# Create agent with robot tool
agent = Agent(tools=[robot, gr00t_inference])

# Start GR00T inference service
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/path/to/checkpoint",
    port=8000,
    data_config="so100_dualcam"
)

# Control robot with natural language
agent("Use my_arm to pick up the red block using GR00T policy on port 8000")

# Cleanup
agent.tool.gr00t_inference(action="stop", port=8000)
```

## Features

- **Robot Support** - LeRobot integration (SO-100/101, Fourier GR-1, Unitree G1, Panda)
- **Policy Abstraction** - Clean interface for any VLA provider (GR00T, ACT, etc.)
- **Multi-Camera** - OpenCV and RealSense support with batch capture
- **Serial Communication** - Feetech servo control and monitoring
- **Strands Tools** - Camera, inference service, and serial tools included

## Architecture

```
strands-robots/
├── robot.py              # Universal Robot class
├── policies/             # Policy abstraction
│   ├── __init__.py       # Policy base + factory
│   └── groot/            # GR00T implementation
│       ├── client.py     # ZMQ client
│       └── data_config.py # 6 embodiment configs
└── tools/                # Strands tools
    ├── gr00t_inference.py  # Docker service manager
    ├── lerobot_camera.py   # Camera operations
    └── serial_tool.py      # Serial communication
```

## Supported Robots

- **SO-100/SO-101** - Single camera or dual camera modes
- **Fourier GR-1** - Bimanual humanoid arms
- **Unitree G1** - Humanoid robot platform
- **Bimanual Panda** - Dual Franka Emika arms
- Any LeRobot-compatible robot

## Policy Providers

```python
from strands_robots import create_policy

# GR00T policy
policy = create_policy(
    provider="groot",
    data_config="so100_dualcam",
    host="localhost",
    port=8000
)

# Mock policy (testing)
policy = create_policy(provider="mock")
```

## Camera Tool

```python
# Discover cameras
agent.tool.lerobot_camera(action="discover")

# Capture single image
agent.tool.lerobot_camera(
    action="capture",
    camera_id="/dev/video0",
    save_path="./captures"
)

# Batch capture from multiple cameras
agent.tool.lerobot_camera(
    action="capture_batch",
    camera_ids=["/dev/video0", "/dev/video2"],
    async_mode=True
)

# Record video
agent.tool.lerobot_camera(
    action="record",
    camera_id="/dev/video0",
    capture_duration=10.0
)
```

## GR00T Inference Service

```python
# Start service
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/model",
    port=8000,
    data_config="so100_dualcam",
    denoising_steps=4
)

# Check status
agent.tool.gr00t_inference(action="status", port=8000)

# List all services
agent.tool.gr00t_inference(action="list")

# Stop service
agent.tool.gr00t_inference(action="stop", port=8000)
```

## Serial Communication

```python
# List available ports
agent.tool.serial_tool(action="list_ports")

# Control Feetech servo position
agent.tool.serial_tool(
    action="feetech_position",
    port="/dev/ttyACM0",
    motor_id=1,
    position=2048  # Center position
)

# Monitor serial data
agent.tool.serial_tool(
    action="monitor",
    port="/dev/ttyACM0",
    baudrate=1000000
)
```

## Custom Robot Configuration

```python
from lerobot.robots.config import RobotConfig

# Use LeRobot config directly
config = RobotConfig(...)
robot = Robot(tool_name="custom", robot=config)

# Or pass LeRobot Robot instance
from lerobot.robots.robot import Robot as LeRobotRobot
lerobot_instance = LeRobotRobot(...)
robot = Robot(tool_name="custom", robot=lerobot_instance)
```

## Data Configs

Available GR00T data configurations:

- `so100` - Single camera SO-100
- `so100_dualcam` - Dual camera SO-100 (front + wrist)
- `so100_4cam` - Quad camera SO-100
- `fourier_gr1_arms_only` - Fourier GR-1 arms
- `bimanual_panda_gripper` - Bimanual Panda
- `unitree_g1` - Unitree G1 humanoid

## License

Apache-2.0

## Links

- **Repository**: https://github.com/cagataycali/strands-robots
- **Issues**: https://github.com/cagataycali/strands-robots/issues
- **Strands SDK**: https://github.com/strands-agents/strands
