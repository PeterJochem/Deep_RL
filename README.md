# Deep Reinforcement Learning
This repo is a collection of deep reinforcement learning algorithms implemented with Tensorflow and Keras. I think deep reinforcement learning has a lot of promise for robotic manipulation and just find it fascinating. I created a folder for each algorithm I implemented. So far, I have implemented Deep Q Learning (DQN), Double Deep Q Learning (DDQN), and Deep Deterministic Policy Gradients (DDPG). 

# Results 
More details about each algorithm and its results can be found in the README.md for each directory. Here are a few of the highlights so far though.
#### DDPG
The Mujoco Physics Simulator (Multi-Joint Dynamics with Contact) has a few OpenAI gym environments for simple robots with continuous control. I implemented DDPG with the same hyper parameters as the original DDPG [paper](https://arxiv.org/abs/1509.02971) and applied it to the Hopper-V2 and Cheetah environments <br />

Below is a gif of the Hopper's learned policy. A video is available at add link <br />
[![](DDPG/media/hopper_learned_policy.gif)]

Below is a gif of the Cheetah's learned policy. A video is available at add link <br />
[![](DDPG/media/cheetah_learned_policy.gif)]

### DDQN on Atari Environments 
Describe learning from pixels <br />
Add a gif of the learned policy 

