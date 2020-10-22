# More details and troubleshooting at https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/
cd venv 
git clone https://github.com/openai/mujoco-py.git
cd mujoco-py
pip install -e .
source export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/peter/.mujoco/mujoco200/bin # Put this in the .bashrc and then source the .bashrc
