#adapted and copied from https://github.com/eczy/rl-drone-coverage/blob/master/field_coverage_env.py

import gym
import numpy as np
import cv2
import copy
from enum import Enum

class FireEnvironment(gym.Env):

    class FireInfo(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class Agent(object):
        def __init__(self, pos, fov):
            self.pos = pos
            self.fov = fov

    def __init__ (self, data_file_name, height, width, shape, theta, num_agents,  X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, max_steps=1000):
        super().__init__()

        self.fire_data = self.readFireData(data_file_name)

        self.height = height
        self.width = width
        self.shape = shape
        self.theta = np.radians(theta)
        self.num_agents = num_agents
        self.X_MIN = X_MIN
        self.X_MAX = X_MAX
        self.Y_MIN = Y_MIN
        self.Y_MAX = Y_MAX
        self.Z_MIN = Z_MIN
        self.Z_MAX = Z_MAX
        self.max_steps = max_steps
        self.steps = 0

        self.action_space = gym.spaces.Box(-1, +1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.array(shape))

        self.agents = {}
        self.map = None
        self.points = None

        self.reset()


    def readFireData(self, file_name):
        f = open(file_name, "r")
        return_data = []
        while True:
            data = f.readline()
            if not data:
                break
            vals = data.split(",")
            if(len(vals) > 1):
                return_data.append(vals)

        return return_data


    def simStep(self, time_t):
        img = np.zeros((self.height,self.width), np.uint8)
        fire_map = {}
        vals = self.fire_data[time_t]
        fire_info = []
        for j in range(0,len(vals)-1,3):
            x = int(vals[j]) - 19
            y = int(vals[j+1]) - 19

            if str(str(x) + "," + str(y)) in fire_map:
                pass
            else:
               fire_map[str(x) + "," + str(y)] = True
               temp = self.FireInfo(int(x), int(y))
               fire_info.append(temp)

        for t in fire_info:
            img = cv2.circle(img, (t.x, t.y), 1, (255,255,255), -1)

        self.points = np.asarray(np.transpose(np.where(img==255)))
        fire_map.clear()
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return img


    def reset(self):
        self.steps = 0
        agents = {}
        self.map = self.simStep(0)
        for i in range(self.num_agents):
            while True:
                pos = np.random.uniform(self.X_MIN, self.X_MAX), np.random.uniform(self.Y_MIN, self.Y_MAX), np.random.uniform(self.Z_MIN, self.Z_MAX - 1)
                if (pos not in agents.values()) and (self.map[round(pos[0]), round(pos[1])] > 0):
                    break
            agents[i] = self.Agent(pos, self.theta)
        self.agents = agents
        return self.state()


    def state(self):
        observation = []
        for obj in self.agents.values():
            for ob in obj.pos:
                observation.append(ob)
        return observation

    def step(self, actions):
        action_list = []
        for i in range(0, len(actions), 6):
            action_list.append(np.argmax(actions[i:i+6]))

        temp_img = self.map.copy()
        color = (255, 0, 0)
        thickness = -1
        temp = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
        #print("states: ",self.state())
        for drone, action in enumerate(action_list):
            self.move_drone(drone, action)
            fov_alt = round(np.tan(self.theta) * self.agents[drone].pos[2])
            start_point = (round(self.agents[drone].pos[0])-fov_alt, round(self.agents[drone].pos[1])-fov_alt)
            end_point = (round(self.agents[drone].pos[0])+fov_alt, round(self.agents[drone].pos[1])+fov_alt)
            temp = cv2.rectangle(temp, start_point, end_point, (0, 1, 1), -1)
            temp = cv2.circle(temp, (round(self.agents[drone].pos[0]), round(self.agents[drone].pos[1])), 0, color, thickness)

        observation = self.state()
        reward = self.reward()
        success = reward > 0.99
        done = success or (self.steps >= self.max_steps)
        #print("Steps ",self.steps," , self.max_steps:",self.max_steps," , self.steps >= self.max_steps : ",(self.steps >= self.max_steps)," , done: ",done)
        cv2.namedWindow("simulation", cv2.WINDOW_NORMAL)
        temp = cv2.resize(temp, (1000,1000), interpolation = cv2.INTER_AREA)
        cv2.imshow("simulation", temp)
        cv2.waitKey(1)
        self.steps += 1

        return observation, reward, done, {'success!': success}


    def reward(self):
        masks = self.view_masks()
        foi = self.map.astype(int)
        foi_orig = foi.copy()
        coverage = 0
        for i, drone in self.agents.items():
            coverage += np.sum(masks[i].flatten() & foi.flatten())
            foi = foi - (masks[i].flatten() & foi.flatten()).reshape(self.shape[0], self.shape[1])

        return(float(float(coverage)/float(sum(foi_orig.flatten()))))


    def view_masks(self):
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in self.map.shape])
        view_masks = {}
        for i, drone in self.agents.items():
            mask = np.zeros_like(self.map).astype(int)
            x, y, z = drone.pos

            for xc, yc in zip(coordsx.flatten(), coordsy.flatten()):
                x_proj = y_proj = round(np.tan(drone.fov) * z)
                if all([
                    xc > x - x_proj,
                    xc < x + x_proj,
                    yc > y - y_proj,
                    yc < y + y_proj
                ]):
                    mask[xc, yc] = True
            view_masks[i] = mask
        return view_masks


    def move_drone(self, drone, action):
        x, y, z = self.agents[drone].pos

        if action == 0:
            new_pos = max(x - 1, 0), y, z
        elif action == 1:
            new_pos = min(x + 1, self.X_MAX), y, z
        elif action == 2:
            new_pos = x, min(y + 1, self.Y_MAX), z
        elif action == 3:
            new_pos = x, max(y - 1, 0), z
        elif action == 4:
            new_pos = x, y, min(z + 1, self.Z_MAX)
        elif action == 5:
            new_pos = x, y, max(z - 1, 1)
        else:
            raise ValueError(f'Invalid action {action} for agent {drone}')

        if new_pos in set(self.state()):
            return
        self.agents[drone].pos = new_pos


    def render(self):
        temp_img = self.map.copy()
        temp = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
        for drone in range(len(self.agents)):
            fov_alt = round(np.tan(self.theta) * self.agents[drone].pos[2])
            start_point = (round(self.agents[drone].pos[0])-fov_alt, round(self.agents[drone].pos[1])-fov_alt)
            end_point = (round(self.agents[drone].pos[0])+fov_alt, round(self.agents[drone].pos[1])+fov_alt)
            temp = cv2.rectangle(temp, start_point, end_point, (0, 1, 1), -1)
            temp = cv2.circle(temp, (round(self.agents[drone].pos[0]), round(self.agents[drone].pos[1])), 0, (255, 0, 0), -1)

        temp = cv2.resize(temp, (220,220), interpolation = cv2.INTER_AREA)
        cv2.imshow("filled", temp)
        cv2.waitKey(0)


    def print(self, file):
        f =  open(file, 'a+')
        coordsx, coordsy = np.meshgrid(*[np.arange(x) for x in self.map.shape])
        view_masks = {}
        for i, drone in self.agents.items():
            mask = np.zeros_like(self.map).astype(int)
            x, y, z = drone.pos
            x_proj = 0
            y_proj = 0
            for xc, yc in zip(coordsx.flatten(), coordsy.flatten()):
                x_proj = np.tan(drone.fov) * z
                y_proj = np.tan(drone.fov) * z

            #print(i, x, y, z, x_proj, y_proj)
            f.write(str(str(i) + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(x_proj) + " " + str(y_proj)) + "\n")
        f.close()
