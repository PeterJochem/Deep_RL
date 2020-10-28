from gym.envs.registration import register

register(
    id='hopping_robot_pmtg-v0',
    entry_point='gym_hopping_robot_pmtg.envs:HoppingRobot_PmtgEnv',
)
