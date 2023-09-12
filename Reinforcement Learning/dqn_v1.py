from uav_env import UAV
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import random
from collections import deque

import tkinter as tk
import numpy as np
import keyboard as kb
import time

UNIT = 40  # pixels
UAV_H = 4  # grid height
UAV_W = 4  # grid width

class UAV(tk.Tk, object):
    def __init__(self):
        super(UAV, self).__init__()
        self.observation_space = np.zeros((UAV_H * UAV_W,))
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.title('UAV')
        self.geometry(str(UAV_H * UNIT) + 'x' + str(UAV_W * UNIT))
        self._build_UAV()

    def _build_UAV(self):
        self.canvas = tk.Canvas(self, bg='white', 
                                height = UAV_H * UNIT,
                                width = UAV_W * UNIT)
        
        # Create grids
        for column in range(0, UAV_W * UNIT, UNIT):
            x0, y0, x1, y1 = column, 0, column, UAV_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for row in range(0, UAV_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, UAV_W * UNIT, row
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])
        
        # hell 1
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black'
        )
        # hell2
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create end
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')
        
        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def manual(self):
        self.bind_all("<Key>", self.key_press)

    def key_press(self, event):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if event.keysym == 'Up':
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif event.keysym == 'Left':
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif event.keysym == 'Down':
            if s[1] < (UAV_H - 1) * UNIT:
                base_action[1] += UNIT
        elif event.keysym == 'Right':
            if s[0] < (UAV_W - 1) * UNIT:
                base_action[0] += UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.oval):
            self.reset()
            print("Game Clear !!!")
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            self.reset()
            print("Game Over !!!")
        else:
            print(s_)


    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )

        # return observation
        return self.canvas.coords(self.rect)
        

    def step(self,action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (UAV_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (UAV_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

         # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False



        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

class Net(nn.Module):
    def __init__(self, _input_size: int, _output_size: int, _hidden_size: int = 50):
        super(Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(_input_size, _hidden_size),
            nn.ReLU(),
            nn.Linear(_hidden_size, _output_size)
        )

    def forward(self, x):
        return self.layers(x)


class DQN:
    def __init__(self, observation_space: int, action_space: int, learning_rate: float = 0.01, gamma: float = 0.99):
        self.net = Net(observation_space, action_space)
        self.target_net = Net(observation_space, action_space)
        self.target_net.load_state_dict(self.net.state_dict())
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=learning_rate)
        self.gamma = gamma

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        predicted_qvalues = self.net(state)

        predicted_qvalues_for_actions = predicted_qvalues[range(len(predicted_qvalues)), action]

        predicted_next_qvalues = self.target_net(next_state)
        target_qvalues_for_actions = reward + self.gamma * torch.max(predicted_next_qvalues, 1)[0]
        target_qvalues_for_actions = torch.where(torch.tensor(done, dtype=torch.bool), reward, target_qvalues_for_actions)

        loss = self.loss_fn(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_action(self, state, epsilon=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        qvalues = self.net(state).detach().numpy()

        if np.random.random() < epsilon:
            action = np.random.choice(len(qvalues[0]))
        else:
            action = np.argmax(qvalues)

        return action
    
def coords_to_state(coords):
    state = np.zeros((UAV_H * UAV_W,))
    state[int(coords[0]) // UNIT + int(coords[1]) // UNIT * UAV_W] = 1
    return state.tolist()




def train_DQN(env, agent, episodes, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64):
    replay_memory = deque(maxlen=10000)
    epsilon = epsilon_start
    episode_rewards = []

    for episode in range(episodes):
        state = env.reset()
        state = coords_to_state(state)  # Convert coordinates to state representation
        done = False
        episode_reward = 0

        while not done:
            # agent takes action
            action = agent.get_action(state, epsilon)

            # apply the action to the environment
            next_state, reward, done = env.step(action)
            next_state = coords_to_state(next_state) if next_state != 'terminal' else None  # Convert coordinates to state representation

            # add to replay memory
            replay_memory.append((state, action, reward, next_state, done))

            # move to next state
            state = next_state
            episode_reward += reward

            # render the environment
            env.render()

            # if enough experiences are collected, update network
            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)

                states = np.array([experience[0] for experience in batch])
                actions = np.array([experience[1] for experience in batch])
                rewards = np.array([experience[2] for experience in batch])
                next_states = np.array([experience[3] if experience[3] is not None else np.zeros_like(states[0]) for experience in batch])  # Handle terminal states
                dones = np.array([experience[4] for experience in batch])

                agent.update(states, actions, rewards, next_states, dones)

        # update target network every 10 episodes
        if episode % 10 == 0:
            agent.target_net
            agent.target_net.load_state_dict(agent.net.state_dict())

        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # append reward for this episode
        episode_rewards.append(episode_reward)

        print("Episode: {}, total reward: {}".format(episode, episode_reward))

    return episode_rewards

env = UAV()

observation_space = 16
action_space = 4

dqn_agent = DQN(observation_space, action_space)
rewards = train_DQN(env, dqn_agent, episodes=1000)





