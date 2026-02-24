---
layout: default
title: Status
---
## Project Summary
Our project’s aim is to develop and train a multi-agent system to play the game of Haxball, a simplified soccer simulation game. We want to train agents so that they can score goals, defend goals, and coordinate with teammates. The agents will be given the current game state as an input, including the position and velocities of all players and the ball, and output an action where they either move in a cardinal direction or attempt to kick the ball. Agents will be trained in multiple scenarios such as 1v0, 1v1, 2v2, and 3v3 matches.

## Approach
Our project uses Proximal Policy Optimization (PPO) to train our agents. The loss function of PPO is shown as follows where $$\epsilon$$ is the clip_range hyperparameter that roughly depicts how far away the new policy can go from the old one.
(Source: https://spinningup.openai.com/en/latest/algorithms/ppo.html)
$$L(a, s, \theta_{k}, \theta) = min(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{k}}(a|s)} A^{\pi_{\theta_{k}}}(s, a), clip(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{k}}(a|s)}, 1 - \epsilon, 1 + \epsilon) A^{\pi_{\theta_{k}}}(s, a))$$

We are using the MlpPolicy along with the default hyperparameters for PPO: learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2. (Source: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

Each player observes the position and velocity of the ball and all players, the current game state (Kickoff or Playing), and the team kicking off. The actions players can take are moving left or right, moving up or down, and kicking the ball.

We currently use 4 classical agents for training against. They are ChaseBot, RandomBot, GoalkeeperBot, and StrikerBot. The ChaseBot simply chases after the ball and kicks it when it is close. The RandomBot takes random actions. The GoalkeeperBot is an advanced bot that acts as a goalkeeper. The StrikerBot is a custom bot we programmed to move behind the ball and kick it towards the goal.

Our current reward function is calculated based on 3 parts: the EventReward, the VelocityPlayerToBallReward, and the VelocityBallToGoalReward. The weights of each reward are 50, 1, and 1 respectively. The EventReward is computed based on the agent touching a ball, kicking a ball, scoring a goal, and conceding a goal. The weights for each part of the EventReward are 0.1, 0.1, 100, and -100 respectively. We implement this with reward function utilities from the HaxballGym library.

## Evaluation
For quantitative evaluation, we measure the performance of different types of agents by looking at their average expected rewards after training over 65536 timesteps. We use Tensorboard to visualize the graphs of these results. We also have each agent compete in a tournament to evaluate how well they are doing. Agents are divided into red and blue teams and are rated based on the final score of each match.

[Insert Tensorboard figure of ep_rew_mean]

For qualitative analysis, we visualize the results externally by reviewing game replays. The Ursinaxball library comes with a way to record the games that are played and save it as a file. We currently have a game replay of an agent…

[Insert video of a game replay]

## Remaining Goals and Challenges
For the remainder of the quarter, we intend to migrate to the PettingZoo library. Currently, we are training our agents in a single-agent environment with Gymnasium. In order to implement self-play for better performance, we will need to use PettingZoo since they are more suitable for multi-agent reinforcement learning.

We hope to create an advanced all-rounder bot that can score and defend goals, as well as coordinate with teammates via passing. We may also need to improve our reward function and optimize training hyperparameters.

A challenge we may face is time constraints. It might take a lot of time to test different algorithms with different hyperparameters in order to create the best possible model.

## Resources Used
HaxballGym and Ursinaxball: For the game environment and physics engine.

stable-baselines3: For the PPO algorithm implementation.

PettingZoo: For the multi-agent environment API.

SuperSuit: For vectorizing environments and handling observations in a multi-agent context.

OpenSkill: For calculating Elo/TrueSkill ratings during evaluation tournaments.

Tensorboard: For visualizing training metrics.