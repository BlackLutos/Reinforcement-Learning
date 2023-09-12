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
    
def update():
    pass



if __name__ == '__main__':
    env = UAV()
    env.manual()
    env.mainloop()