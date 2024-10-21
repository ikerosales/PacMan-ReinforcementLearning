# Pacman Q-Learning Project
**Authors**: Iker Rosales & Enrique Colmenarejo

![image](https://github.com/user-attachments/assets/e1c4e578-0123-4422-ade0-3f106fc94dc8)


## Overview
This repository contains the implementation of a Q-learning algorithm applied to the classic Pac-Man game. The goal is to enable Pac-Man to learn an optimal action-selection policy that maximizes long-term rewards while navigating through various game environments. The project outlines the development of state selection strategies, reward functions, and the Q-table updating process that together enhance the agent's learning efficiency.

## Project Structure
- **/PACMAN_RL**: Contains the main Q-learning implementation.
- **/PACMAN_RL/layouts**: Includes various game maps used for training the Pac-Man agent.
- **/PACMAN_RL/q-table.txt**: Store the Q-table generated during learning
- **README.md**: This file.

## Phase 1: State Selection
The initial task involved selecting the states necessary for achieving optimal learning outcomes. Various approaches were evaluated:

1. **Initial State Selection**: We began with a simple quadrant division of the map using two functions \( f: y = x \) and \( g: y = -x \), creating four regions: North, South, East, and West. However, this proved inadequate.

2. **Expanded State Space**: We iterated on the initial approach by considering additional states that accounted for wall positions:
   - One wall in each direction.
   - Two walls leading to six new states.
   - Three walls.

   This brought our total to 15 possible states. Although this was a step forward, the agent's learning was inconsistent across different maps.

3. **Quadrant-Based Approach**: We ultimately adopted a method based on map quadrants, complemented by creating states along the axes (x=0 and y=0). This adjustment significantly improved learning efficiency.

4. **Distance Function**: To further enhance state representation, we integrated a function that calculates the distance to ghosts, taking walls into account, resulting in a total of 120 states.

## Phase 2: Implementation in PyCharm
We implemented the above state selection strategies within the Q-learning class, specifically in the `computePosition` method. This method extracts Pac-Man's position and the nearest ghost's position to effectively categorize the map into eight regions based on relative positioning. We also considered wall positions to determine the index of the Q-table.

### Reward Function
The reward system is critical for guiding the agent's learning. Our strategy includes:

- High rewards for eating pac-dots.
- Penalties for illegal actions and distances from ghosts.
- A mechanism to track visited positions, penalizing repeated visits during ghost pursuits.

We discovered that utilizing the `getDistance()` method from `self.distancer` allowed us to consider wall positions when calculating distances, contributing to successful convergence across different maps.

## Q-table Updating
The Q-table is updated using the stochastic update equation:

Q(s, a) ← Q(s, a) + α [ R + γ max_a Q(s', a) - Q(s, a) ]

Where:
- Q(s, a) is the Q-value for state s and action a.
- R is the reward received after taking action a.
- α (learning rate) determines the weight of new information versus previous Q-value.
- γ (discount factor) reflects the importance of future rewards.

This iterative process ensures that the Q-values converge to reflect the agent's learned experience over time.

## Conclusion
The project showcases the application of Q-learning in a dynamic environment like Pac-Man, emphasizing the importance of state selection, reward design, and Q-table management. While the implemented strategies proved effective, there is potential for further refinement to enhance learning efficiency and adaptability to various map configurations.
