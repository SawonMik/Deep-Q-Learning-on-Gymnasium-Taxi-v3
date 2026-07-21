# Deep-Q-Learning-on-Gymnasium-Taxi-v3
Q-learning agent trained on OpenAI Gymnasium's Taxi-v4 environment using a tabular Q-table approach. Includes epsilon-greedy exploration with decay, reward tracking, and model persistence via pickle.
# Taxi-v4 Q-Learning Agent

A tabular Q-learning implementation that trains an agent to solve Gymnasium's `Taxi-v4` environment — picking up a passenger and dropping them at the correct destination in a 5x5 grid world.

## Overview

This project uses classic Q-learning (no neural networks) to solve a discrete-state, discrete-action reinforcement learning problem. The agent learns an optimal policy by updating a Q-table over many training episodes using the Bellman equation.

## Environment

- **Task**: [Taxi-v4](https://gymnasium.farama.org/environments/toy_text/taxi/) — navigate a taxi to pick up a passenger and drop them off at one of four designated locations.
- **State space**: 500 discrete states (25 taxi positions × 5 passenger locations × 4 destinations)
- **Action space**: 6 discrete actions (south, north, east, west, pickup, dropoff)
- **Rewards**: -1 per step, +20 for successful dropoff, -10 for illegal pickup/dropoff

## Algorithm

- **Method**: Q-learning (off-policy temporal difference learning)
- **Exploration**: Epsilon-greedy strategy, starting at ε=1 (fully random) and decaying linearly to 0
- **Update rule**: Standard Bellman equation update
