# Twin Delayed DDPG
This is an implementation of [Twin Delayed Deep Deterministic Policy Gradients](https://arxiv.org/abs/1802.09477) (TD3 DDPG or simply TD3) in Tensorflow. 


# Challenges Addressed By TD3
TD3 is built on top of the work of [DDPG](https://arxiv.org/abs/1509.02971). It addresses the instability of DDPG and its tendency to over-estimate Q-values. TD3 uses three new methods to mitigate these problems of DDPG. 

## Method 1: Clipped Double Q-Learning 
TD3 learns two Q-functions and when computing target values, uses the minimum value of the two Q-functions. This helps prevent the agent from overestimating the Q-values of certain (state, action) pairs. 

## Method 2: Delayed Policy Updates 
TD3 updates the actor, or policy network, less often than the critic, or value networks. This helps stabilize the training process. The TD3 paper linked to above has more derivations about why this is true. 

## Target Policy Smoothing
TD3 adds noise to the target actions in order to smooth out the Q-functions across the action inputs. This makes the learned policy more robust to small changes in actions. DDPG adds noise online when we take an action and then records the noisy action taken into the replay buffer. This noise encourages the agent to explore. TD3 has this same exploratory noise but also adds noise to the actions when computing targets during training. This noise makes it much harder for the agent to exploit errors in the Q-function. DDPG often develops incorrect, sharp peaks in the value function for certain actions. Smoothing out the action values helps prevent the Q-function from developing these peaks. 

# TD3 Psuedocode
![](media/TD3_Pseudocode.png)

# Results 
Wow! The agent does learn much faster than the DDPG agent. 