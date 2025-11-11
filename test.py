#!/usr/bin/env python3
from strands import Agent
from strands_robots import Robot

robot = Robot(
    tool_name="orange_arm",
    robot="so101_follower",
    cameras={
        "wrist": {"type": "opencv", "index_or_path": "/dev/video0", "fps": 30},
        "front": {"type": "opencv", "index_or_path": "/dev/video2", "fps": 30},
    },
    port="/dev/ttyACM0",
    data_config="so100_dualcam",
)

agent = Agent(tools=[robot], load_tools_from_directory=True)

print("🧠 Starting GR00T inference service...")
agent.tool.gr00t_inference(
    action="start",
    checkpoint_path="/data/checkpoints/gr00t-wave/checkpoint-300000",
    port=8000,
    data_config="so100_dualcam",
)
print("✅ GR00T service ready!")


print("🤖 Testing robot movement...")
# Policy specified in the request - agent will pass to robot tool
result = agent("Use the orange_arm robot to wave the arm using GR00T policy on port 8000")
print(f"✅ Result: {result}")

# 🧹 CLEANUP: Stop GR00T service
print("🧹 Stopping GR00T service...")
agent.tool.gr00t_inference(action="stop", port=8000)
print("✅ Cleanup complete!")
