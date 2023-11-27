# Introduction
* Environment for single agent
* Common environments:
	* Cart Pole
	* Pendulum
	* Mountain car
	* Mujoco
	* Atari
* Key Functions
	* make
	* reset
	* step
	* render
* You can able create own environment with the gymnasium api
# Code
## Setup the environment
``` python
# import the package
import gymnasium as gym

# create an environment
env = gym.make("environment_name")

# init the env
observation,info = env.reset()

# action
for _ in range(1000):
	action = env.action_space.sample()
	observation,reward,terminated,truncated,info = env.step(action)

	if terminated or truncated:
		observation,info = env.reset()

# close the env
env.close()
```
## Evaluation
``` python
# create a new environment for evaluation
eval_env = gym.make("environment_name")
mean_reward,std_reward = evaluate_policy(model,eval_env,n_eval_episodes = 10 ,deterministic=True)

print(f"Mean reward : {mean_reward} +/- {std_reward}")
```
# Resource
1. [Gymnasium Documentation (farama.org)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)