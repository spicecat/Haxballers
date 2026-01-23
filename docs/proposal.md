---
layout: default
title: Proposal
---

## Summary

Our project idea is to develop and train a multi-agent system to play the game of Haxball, a simplified soccer simulation game. The behaviors we want the agents to learn is to score goals, defend against goals, and pass to teammates. Agents will use the current game state as input, including the positions and velocities of all players and the ball, and output a cardinal direction to move in or a kick action.

## Project Goals

- Minimum Goal: Develop an agent that can move to the ball.
- Realistic Goal: Develop an agent that can score and defend goals.
- Moonshot Goal: Develop an agent that can coordinate with teammates by passing the ball.

## Algorithms

We anticipate using model-free on-policy multi-agent reinforcement learning to train the Haxball agents. The training environment will progress through the following stages: 1v0, 1v1, 2v1, 2v2, and 3v3.

## Evaluation Plan

For quantitative evaluation, we will have agents at different levels of training compete in an Elo rating system. Some metrics we may measure are the number of games won, the number of goals scored, the number of goals defended, and pass frequency. As a baseline approach, we will train agents to move behind the ball and kick it towards the opposing goal. We estimate that successful training will improve the win rate metric by 90%.

For qualitative analysis, we will use the 1v0 training environment for sanity checks. We will visualize the results externally by reviewing game replays. A successful result is expected to display agents moving efficiently, kicking the ball towards the opposing goal, and passing the ball towards open teammates.

## AI Usage

AI was used for coding assistance.
