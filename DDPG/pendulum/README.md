# Penduluum Environment
I implemented DDPG to work for the OpenAI gym's Pendulum environment where the agent must learn to invert and balance a pendulum. The agent can apply a torque on the joint at each time step to make the pendulum achieve its goal. The control signal is continuous which is a big change from the other Deep RL approaches I have worked on so far. I used this environment as a validation step on my way to implementing DDPG for higher dimensional tasks. The agent should be able to learn how to invert the penduluum in a few minutes. The MuJoCo environments take quite a while to learn so this is a easy way to test DDPG before running a longer training session in MuJoCo. Below is a gif of the agent's learned policy. <br />

[![](media/learned_policy_pendulum.gif)]

