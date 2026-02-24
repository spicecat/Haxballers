---
layout: default
title: Status
---
## Video Summary
<div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%;
 padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
 border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
    src="https://www.canva.com/design/DAHCNPmejgo/NItgT2w0f0EnpV4Lfk27LQ/watch?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>
<a href="https:&#x2F;&#x2F;www.canva.com&#x2F;design&#x2F;DAHCNPmejgo&#x2F;NItgT2w0f0EnpV4Lfk27LQ&#x2F;watch?utm_content=DAHCNPmejgo&amp;utm_campaign=designshare&amp;utm_medium=embeds&amp;utm_source=link" target="_blank" rel="noopener">Haxballers Status Report</a>

## Project Summary
Our project’s aim is to develop and train a multi-agent system to play the game of Haxball, a simplified soccer simulation game. We want to train agents so that they can score goals, defend goals, and coordinate with teammates. The agents will be given the current game state as an input, including the position and velocities of all players and the ball, and output an action where they either move in a cardinal direction or attempt to kick the ball. Agents will be trained in multiple scenarios such as 1v0, 1v1, 2v2, and 3v3 matches.

## Approach
Our project uses Proximal Policy Optimization (PPO) to train our agents via the Stable Baselines3 library. The loss function of PPO is shown as follows where $$\epsilon$$ is the clip_range hyperparameter that roughly depicts how far away the new policy can go from the old one.

$$L(a, s, \theta_{k}, \theta) = min(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{k}}(a|s)} A^{\pi_{\theta_{k}}}(s, a), clip(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{k}}(a|s)}, 1 - \epsilon, 1 + \epsilon) A^{\pi_{\theta_{k}}}(s, a))$$

We are using the MlpPolicy along with the default hyperparameters for PPO: 
- learning_rate = 0.0003
- n_steps = 2048
- batch_size = 64
- n_epochs = 10
- gamma = 0.99
- gae_lambda = 0.95
- clip_range = 0.2

Each player observes the position and velocity of the ball and all players, the current game state (Kickoff or Playing), and the team kicking off. The actions players can take are moving left or right, moving up or down, and kicking the ball.

We currently use 4 classical agents for training against. They are ChaseBot, RandomBot, GoalkeeperBot, and StrikerBot. The ChaseBot simply chases after the ball and kicks it when it is close. The RandomBot takes random actions. The GoalkeeperBot is an advanced bot that acts as a goalkeeper. The StrikerBot is a custom bot we programmed to move behind the ball and kick it towards the goal.

Our current reward function is calculated based on 3 parts: the EventReward, the VelocityPlayerToBallReward, and the VelocityBallToGoalReward. The weights of each reward are 50, 1, and 1 respectively. The EventReward is computed based on the agent touching a ball, kicking a ball, scoring a goal, and conceding a goal. The weights for each part of the EventReward are 0.1, 0.1, 100, and -100 respectively. We implement this with reward function utilities from the HaxballGym library.

## Evaluation
For quantitative evaluation, we measure the performance of different types of agents by looking at their average expected rewards after training over 65536 timesteps. We use Tensorboard to visualize the graphs of these results. 

[Insert Tensorboard figure of ep_rew_mean]

We also implemented an Elo rating system using the Plackett-Luce model from the openskill library. We run a tournament where our trained agents compete against the scripted bots (ChaseBot, GoalkeeperBot) to determine a relative skill rating.

For qualitative analysis, we visualize the results externally by reviewing game replays. The Ursinaxball library comes with a way to record the games that are played and save it as a file. We currently have a game replay of an agent…
[Insert video of a game replay]

## Remaining Goals and Challenges
For the remainder of the quarter, we intend to finish migrating to the PettingZoo library. Currently, we are training our agents in a single-agent environment against classical bots with Gymnasium. In order to implement self-play for better performance, we will need to use PettingZoo since it is more suitable for multi-agent reinforcement learning.

We hope to create an advanced all-rounder bot that can score and defend goals, as well as coordinate with teammates via passing. We may also need to improve our reward function and optimize training hyperparameters.

A challenge we may face is time constraints. It may take a lot of time to test different algorithms with different hyperparameters in order to create the best possible model.
Another challenge we may face is playing models that were trained in different environments against each other. Normalizing observations in a new environment to fit model’s expected observations shape may affect model performance.

## Resources Used
[**Ursinaxball**](https://github.com/HaxballGym/Ursinaxball): Haxball physics engine

[**HaxballGym**](https://github.com/HaxballGym/HaxballGym): Gymnasium environment for ursinaxball

[**stable-baselines3**](https://stable-baselines3.readthedocs.io/en/master/): PPO algorithm implementation

[**PettingZoo**](https://pettingzoo.farama.org/): Multi-agent environment

**SuperSuit**: Vectorize Gymnasium environment for PettingZoo’s multi-agent context

**OpenSkill**: Calculate model ratings during evaluation tournaments

**Tensorboard**: Visualize training metrics

**Gemini 3**: Code debugging and assistance

**Citations:**
- **PPO Algorithm Formula**: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>

- **PPO Default Hyperparameters**: <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>

