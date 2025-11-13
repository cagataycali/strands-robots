from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="orange_follower",
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="orange_leader",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)


# python -m lerobot.teleoperate \
#     --robot.type=so101_follower \
#     --robot.port=/dev/ttyACM0 \
#     --robot.id=my_awesome_follower \
#     --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
#     --teleop.type=so101_leader \
#     --teleop.port=/dev/ttyACM1 \
#     --teleop.id=my_awesome_leader
