# Lunar Lander

>[!INFO]-
>* Always start with the documentation of the environment

## Environment
For lunar lander environment we use the [[Gymnasium]] library

There are two versions of environments:
* Discrete - Only two states (engine off or engine on)
* Continuous - It contains many values like engine speed ranging from 0 to 100 km/hours
### Action Space

There are 4 actions:

|Number | Action|
|---|---|
|0|Do nothing|
|1|Fire the left engine|
|2|Main Engine on|
|3|Fire the right engine|

### Observation Space

* 8 Dimensional vector space
	* Coordinates of `x` and `y`
	* Linear velocity of `x` and `y`
	* Angle
	* Angular velocity
	* Boolean 1 - is right leg is in contact to the moon?
	* Boolean 2 - is left leg is in contact to the moon?

### Rewards

>What is episode?

Reward is based on the following criteria :
* Distance to the landing pad
* Speed of the lander
* Angle when the lander is landed
* If a leg is in contact to the ground then the reward is 10 points
* If a side engine fires then 0.03 points is reduced per frame
* If a main engine fire then 0.3 points is reduced per frame
* Crashing points is -100
* Landing point is +100
* At least a 200 points is considered as a episode

### Episode termination

* The lander crashes
* Lander lands on outside of the post
* The lander is not awake ~ the lander stops moving or it doesn't interact with the environment

### Vectorized Environment

Vectorized environment is a method in RL that allows stacking multiple individual environments to single environment. It increases the diversity of experiences during training of an agent.

## Model

The `Stable Baseline (SB3)` library is used for the model.
In this we use `PPO (Proximal Policy Optimization)` RL algorithm. PPO is combination of `value`+`policy` based RL method. The following are the steps:
1. Create an environment using `gymnasium`
2. Instantiate the model `model = PPO("MlpPolicy")`
3. Train the model with `model.learn()` method

### Arguments

|Argument|Description|
|---|---|
|policy|`MlpPolicy` is a multiple layer perceptron policy. The other policies are `CnnPolicy - grid like data (images)`,`ActorCriticPolicy - used to create custom policy`|
|env|environment|
|n_steps|The number of steps to run for each environment per update. This parameter controls how much data is collected before each update of the policy network.
|batch_size|The size of a batch of data when optimizing the policy network. This parameter affects how many gradient steps are taken per update.
|n_epochs|Number of epochs|
|gamma|Discount factor. Value close to 1 is long-term and close to 0 is short term|
|gae-lambda|GAE (Generalized Advantage Estimation) is a technique to reduce the variance of the policy gradient by using a combination of the rewards and the value function to estimate the advantage function. A value close to 1 means more bias but less variance, while a value close to 0 means less bias but more variance
|ent_coef| entropy coefficient, it regulates the Exploration-Exploitation Tradeoff. Higher value means more exploration and less exploitation 
|clip_range| This factor maintains the ratio of the old policy and new policy. Lower value is less deviation and more stability 
|verbose| level of info printed at training, 0 - no output , 1 - some output, 2 - more output

## Code

``` python
# importing required library
import gymnasium as gym
from stable_baseline3 import PPO
from stable_baseline3.common.evaluation import evaluate_policy

# Step 1 : Create a training environment
env = gym.make("LunarLander-v2")
obs,info = env.reset() # init the environment

# Step 2 : Create an agent
model = PPO(
			policy = "MlpPolicy",
			env = env,
			n_steps = 1024,
			batch_size = 64,
			n_epochs = 10,
			gamma = 0.98,
			gae_lambda = 0.998,
			ent_coef = 0.01,
			verbose = 2
)

# Step 3 : Train the model
model.learn(total_timesteps = 2000000,progress_bar = True)
model.save("lunar lander")

# Step 4 : Evaluate the model
eval_env = gym.make("LunarLander-v2")
mean,std = evaluate_policy(model,eval_env,n_eval_episodes=10,deterministic=True)
print(f"Mean : {mean} \n Std : {std}")
```

## Resource
1. [Lunar Lander Environment - Gymnasium Documentation (farama.org)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
2. [Lunar Lander Code 🚀 - Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit1/hands-on)
3. [Lunar Landed Model Card by Me 🚀](https://huggingface.co/JaiSurya/PPO-LunarLander-v2/tree/main/)