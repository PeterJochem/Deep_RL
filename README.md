# Deep Reinforcement Learning
This repo is a collection of deep reinforcement learning algorithms implemented with Tensorflow and Keras. I think deep reinforcement learning has a lot of promise for robotic manipulation and just find it fascinating. I created a folder for each algorithm I implemented. So far, I have implemented Deep Q Learning (DQN), Double Deep Q Learning (DDQN), and Deep Deterministic Policy Gradients (DDPG). I also created a custom OpenAI gym environment of a hopping robot on soft ground. 


# Results 
More details about each algorithm and its results can be found in the README.md for each directory. Here are a few of the highlights so far though.
## DDPG
The Mujoco Physics Simulator (Multi-Joint Dynamics with Contact) has a few OpenAI gym environments for simple robots with continuous control. I implemented DDPG with the same hyper parameters as the original DDPG [paper](https://arxiv.org/abs/1509.02971) and applied it to the Hopper-V2 and Cheetah environments. The Hopper and Cheetah environments feature robots who must learn what torques to apply to their motors in order to produce forward translation. <br />

Below is a gif of the Hopper's learned policy. A video is available [here](https://youtu.be/E0tvLX5sxv0?t=281) <br />
![](DDPG/media/hopper_learned_policy.gif)

Below is a gif of the Cheetah's learned policy. A video is available [here](https://youtu.be/DQCQSEspLhs) <br />
![](DDPG/media/cheetah2.gif)


### Custom OpenAI Gym Environment in PyBullet
I created my own custom gym environment in PyBullet. I wanted to simulate a hopping robot walking on soft ground. PyBullet does not support granular materials in this sense so I simulated the robot's foot interacting with granular materials in Chrono, gathered a dataset, and trained a neural network to map the robot's state to the ground reaction forces and moments of torque. I then used this neural network to apply forces and torques on the robot's foot as it interacted with the bed of granular material. I still need to add a way to visualize where in space the granular material is. Below is an image of the robot in its home position. More details can be found in the h3pper directory. <br />
Custom PyBullet Environment             |  Reward Per Epoch over Time
:-------------------------:|:-------------------------:
![](DDPG/h3pper/gym-hopping_robot/images/start_position.png)  | ![](DDPG/media/pybullet_hopper_progress.png)

## DDQN on Atari Environments 
I am in the process of implementing RL algorithms with convolutional neural networks to run on the Atari gym environments. This is where I will give a brief description and gif of those results. 

