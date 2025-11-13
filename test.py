#!/usr/bin/env python3
from strands import Agent
from strands_tools import shell, editor, file_read, file_write
from strands_robots import Robot, gr00t_inference, lerobot_camera, lerobot_calibrate, lerobot_teleoperate, pose_tool

robot = Robot(
    tool_name="orange_arm",
    robot="so101_follower",
    cameras={
        "wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 15, "fourcc": "MJPG"},
        "front": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 15, "fourcc": "MJPG"},
    },
    port="/dev/ttyACM0",
    data_config="so100_dualcam",
)

agent = Agent(
    tools=[shell, editor, file_read, file_write, robot, gr00t_inference, lerobot_camera, lerobot_calibrate, lerobot_teleoperate, pose_tool],
    load_tools_from_directory=True,
)

agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/gr00t-wave/checkpoint-300000",
    port=8000,
    data_config="so100_dualcam",
)

# agent("Use the orange_arm robot to wave the arm using GR00T policy on port 8000")

# agent.tool.gr00t_inference(action="stop", port=8000)

while True:
    agent(input("\n# "))
