from gym.envs.registration import register

register(
    id='hopping_robot-v0',
    entry_point='gym_hopping_robot.envs:HoppingRobotEnv',
)
