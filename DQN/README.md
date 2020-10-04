# Description
I used OpenAI's cart-pole enviroment to implement a Q-learning agent. I used Tensorflow/Keras to implement the neural network. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

# Results
This is a gif of the learned policy. Pretty stable!
![Learned Policy](https://github.com/PeterJochem/Cart_Pole_RL/blob/master/stable.gif "Learned Policy")

![Agent Learning Policy](https://github.com/PeterJochem/Cart_Pole_RL/blob/master/rewardPerEpisode.png "Agent Learning Policy")

A video of the agent learning from scratch is here: https://youtu.be/YQp4s52MlyM

# Tensorflow and Virtual Enviroment Setup
It is easiest to run Tensorflow from a virtual enviroment on Linux. Here are instructions on how to setup Tensorflow and the virtual enviroment https://linuxize.com/post/how-to-install-tensorflow-on-ubuntu-18-04/

To activate the virtual enviroment: ```source venv/bin/activate```

# How to Run My Code
```python3 cart_pole.py```
