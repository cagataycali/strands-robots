#!/usr/bin/env python3
"""
WORKING Simple Robot Test - Fixed version of simple_test.py
"""

from strands import Agent
from strands_robots import Robot
from tools.gr00t_inference import gr00t_inference

print("🧠 Starting GR00T inference service...")
gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/gr00t-wave/checkpoint-300000",
    port=8000,
    data_config="so100_dualcam",
    policy_name="groot_policy"
)
print("✅ GR00T service ready!")

robot = Robot(
    tool_name="orange_arm",
    robot="so101_follower", 
    cameras={
        "wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30},
        "front": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30}
    },
    port="/dev/ttyACM0",
    data_config="so100_dualcam"
)

robot.register_policy_service(
    "groot", 
    port=8000, 
    provider="groot",
    data_config="so100_dualcam"
)

robot.default_policy_name = "groot"

agent = Agent(tools=[robot], load_tools_from_directory=True)

print("🤖 Testing robot movement...")
result = agent("Wave the arm")
print(f"✅ Result: {result}")

# 🧹 CLEANUP: Stop GR00T service
print("🧹 Stopping GR00T service...")
gr00t_inference(action="stop", port=8000)
print("✅ Cleanup complete!")
print(f"✅ Result: {result}")