import numpy as np
import matplotlib.pyplot as plt

class Gridworld:
    def __init__(self, width=12, height=4):
        self.width = width
        self.height = height
        self.start = (height - 1, 0)
        self.goal = (height - 1, width - 1)
        self.cliff = [(height - 1, i) for i in range(1, width - 1)]
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # Define actions: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.state[0] > 0:  # up
            self.state = (self.state[0] - 1, self.state[1])
        elif action == 1 and self.state[0] < self.height - 1:  # down
            self.state = (self.state[0] + 1, self.state[1])
        elif action == 2 and self.state[1] > 0:  # left
            self.state = (self.state[0], self.state[1] - 1)
        elif action == 3 and self.state[1] < self.width - 1:  # right
            self.state = (self.state[0], self.state[1] + 1)

        reward = -1  # default reward for each step
        if self.state in self.cliff:
            reward = -100  # penalty for falling off the cliff
            self.state = self.start  # reset to start
        elif self.state == self.goal:
            reward = 0  # reward for reaching the goal

        return self.state, reward

    def render(self):
        grid = np.zeros((self.height, self.width))
        grid[self.start] = 1  # Start
        grid[self.goal] = 2  # Goal
        for cliff in self.cliff:
            grid[cliff] = -1  # Cliff
        print(grid)

# Q-learning algorithm

def q_learning(env, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.height, env.width, 4))  # Q-table
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            if np.random.rand() < epsilon:
                action = np.random.choice(4)  # Explore
            else:
                action = np.argmax(Q[state[0], state[1]])  # Exploit

            next_state, reward = env.step(action)
            total_reward += reward
            best_next_action = np.argmax(Q[next_state[0], next_state[1]])
            Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], best_next_action] - Q[state[0], state[1], action])
            state = next_state
            if reward == 0:  # Reached goal
                break

        total_rewards.append(total_reward)
    return Q, total_rewards

# SARSA algorithm

def sarsa(env, episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.height, env.width, 4))  # Q-table
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        if np.random.rand() < epsilon:
            action = np.random.choice(4)  # Explore
        else:
            action = np.argmax(Q[state[0], state[1]])  # Exploit

        total_reward = 0
        while True:
            next_state, reward = env.step(action)
            total_reward += reward
            if np.random.rand() < epsilon:
                next_action = np.random.choice(4)  # Explore
            else:
                next_action = np.argmax(Q[next_state[0], next_state[1]])  # Exploit

            Q[state[0], state[1], action] += alpha * (reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
            state, action = next_state, next_action
            if reward == 0:  # Reached goal
                break

        total_rewards.append(total_reward)
    return Q, total_rewards

# Visualization function

def plot_rewards(q_rewards, sarsa_rewards):
    plt.plot(q_rewards, label='Q-learning')
    plt.plot(sarsa_rewards, label='SARSA')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    env = Gridworld()
    episodes = 500
    q_learning_q, q_learning_rewards = q_learning(env, episodes)
    sarsa_q, sarsa_rewards = sarsa(env, episodes)

    plot_rewards(q_learning_rewards, sarsa_rewards)