#!/bin/env/python
import time
import logging
import random
import numpy as np
import matplotlib.pyplot as plt

import gym
from gym import spaces
from gym.envs.classic_control import rendering


class CatAndMouseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }
    
    def __init__(self, height=8,width=10,mode_obstacle=0,mode_mouse=0,map=None,mouse=None):
        """
        input
            height, width: size of map
            mode_obstacle: 0-obstacles fixed, 1-generate obstacles randomly
            mode_mouse   : 0-mouse static, 1-mouse move randomly
        """
        super(CatAndMouseEnv, self).__init__()
        self.ratio = 0.1
        # generate world
        if map is not None:
            self.world = map
            self.height,self.width = map.shape
            self.mode_obstacle = mode_obstacle
            self.mode_mouse = mode_mouse
            self.mouse = mouse
            self.cat = np.array([np.random.randint(self.height),np.random.randint(self.width)])
            while self.world[self.cat[0],self.cat[1]] or all(self.mouse==self.cat):
                self.cat = np.array([np.random.randint(self.height),np.random.randint(self.width)])
            self.counts = 0 
        else:
            self.height = height
            self.width = width
            self.mode_obstacle = mode_obstacle
            self.mode_mouse = mode_mouse
            self.world = np.random.random((height,width))
            self.world = (self.world < self.ratio).astype(np.int8)
            self.cat = np.array([np.random.randint(self.height),np.random.randint(self.width)])
            while self.world[self.cat[0],self.cat[1]]:
                self.cat = np.array([np.random.randint(self.height),np.random.randint(self.width)])
            self.mouse = np.array([np.random.randint(self.height),np.random.randint(self.width)])
            while self.world[self.mouse[0],self.mouse[1]] or all(self.mouse==self.cat):
                self.mouse = np.array([np.random.randint(self.height),np.random.randint(self.width)])
        #plt.imshow(self.world)

        # action spaces
        #self.action_map = {'up':np.array([-1, 0]), 'right':np.array([0, 1]), 'down':np.array([1, 0]), 'left':np.array([0, -1])}#, 'hold':np.array([0, 0])}
        self.action_map = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
        self.action_space = spaces.Discrete(4) # [1: Up, 2: Right, 3: Down, 4: Left]

        # visualization
        self.viewer = None #rendering.Viewer(np.maximum(20*height,600), np.maximum(20*width,1000))   # 600x400 是画板的长和框
        self.u_size = 15
        
        
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        reward = 0
        
        # cat move
        new_state = self.cat + self.action_map[action]
        new_state_x = np.clip(new_state[0], 0, self.height-1)
        new_state_y = np.clip(new_state[1], 0, self.width-1)
        if new_state_x!=new_state[0] or new_state_y!=new_state[1]:   # out of range
            cat_new = self.cat
            reward -= 10
        elif self.world[new_state_x,new_state_y]:   # obstacle
            cat_new = self.cat
            reward -= 10
        else:
            cat_new = np.array([new_state_x,new_state_y])
            
        # mouse move
        if self.mode_mouse:
            prob = np.ones(len(self.action_map))*0.2
            for i in range(len(self.action_map)):
                rpos = self.mouse-self.cat
                if np.dot(rpos, self.action_map[i]) > 0:
                    prob[i] = 0.8
            prob /= np.sum(prob)
            new_state = self.mouse + self.action_map[np.random.choice(len(self.action_map), p=prob)]
            '''if self.mouse[0]-self.cat[0]:
                away_direc.append(np.array([np.sign(self.mouse[0]-self.cat[0]), 0]))
            if self.mouse[1]-self.cat[1]:
                away_direc.append(np.array([0, np.sign(self.mouse[1]-self.cat[1])]))
            if len(away_direc):
                new_state = self.mouse + random.sample(away_direc,1)[0]
            else:
                new_state = self.mouse'''
            new_state_x = np.clip(new_state[0], 0, self.height-1)
            new_state_y = np.clip(new_state[1], 0, self.width-1)
            if new_state_x!=new_state[0] or new_state_y!=new_state[1]:   # out of range
                pass
            elif self.world[new_state_x,new_state_y]:   # obstacle
                pass
            else:
                self.mouse = np.array([new_state_x,new_state_y])
            
        self.cat = cat_new
        self.counts += 1
        
        # reward
        done = all(self.cat==self.mouse)
        if not done:
            reward -= 1
        else:
            reward += 100
            
        return (self.cat, self.mouse), reward, done, {}
        
        
    def reset(self):
        # reset map
        if self.mode_obstacle:
            self.world = np.random.random((self.height,self.width))
            self.world = (self.world < self.ratio).astype(np.int8)
            
        # reset cat and mouse
        self.cat = np.array([np.random.randint(self.height),np.random.randint(self.width)])
        while self.world[self.cat[0],self.cat[1]]:
            self.cat = np.array([np.random.randint(self.height),np.random.randint(self.width)])
        if self.mode_mouse:
            self.mouse = np.array([np.random.randint(self.height),np.random.randint(self.width)])
            while self.world[self.mouse[0],self.mouse[1]] or all(self.mouse==self.cat):
                self.mouse = np.array([np.random.randint(self.height),np.random.randint(self.width)])
        self.counts = 0 
        
        return (self.cat, self.mouse)
        

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        u_size = self.u_size
        m = 1       # gaps between two cells

        # 如果还没有设定屏幕对象，则初始化整个屏幕具备的元素。
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.u_size*self.width, self.u_size*self.height)

            # draw cells
            for x in range(self.width):
                for y in range(self.height):
                    v = [(x*u_size+m, y*u_size+m),
                         ((x+1)*u_size-m, y*u_size+m),
                         ((x+1)*u_size-m, (y+1)*u_size-m),
                         (x*u_size+m, (y+1)*u_size-m)]

                    rect = rendering.FilledPolygon(v)
                    if self.world[y,x]:
                        rect.set_color(0.6,0.6,0.6)
                    else:
                        rect.set_color(1.0,1.0,1.0)
                    '''r = self.grids.get_reward(x,y)/10
                    if r < 0:
                        rect.set_color(0.9-r, 0.9 + r, 0.9 + r)
                    elif r > 0:
                        rect.set_color(0.3, 0.5 + r, 0.3)
                    else:
                        rect.set_color(0.9,0.9,0.9)'''
                    self.viewer.add_geom(rect)
                    # 绘制边框, draw frameworks
                    v_outline = [(x*u_size+m, y*u_size+m),
                                     ((x+1)*u_size-m, y*u_size+m),
                                     ((x+1)*u_size-m, (y+1)*u_size-m),
                                     (x*u_size+m, (y+1)*u_size-m)]
                    outline = rendering.make_polygon(v_outline, False)
                    outline.set_linewidth(3)
                        
            # draw cat
            self.agent = rendering.make_circle(u_size/2, 30, True)
            self.agent.set_color(0.8, 0.9, 0.5)
            self.viewer.add_geom(self.agent)
            self.agent_trans = rendering.Transform()
            self.agent.add_attr(self.agent_trans)

            # draw mouse
            self.target = rendering.make_polygon([(-13,-11),(13,-11),(0,11)])
            self.target.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(self.target)
            self.target_trans = rendering.Transform()
            self.target.add_attr(self.target_trans)

        # update position of an agent
        self.agent_trans.set_translation((self.cat[1]+0.5)*u_size, (self.cat[0]+0.5)*u_size)     
        self.target_trans.set_translation((self.mouse[1]+0.5)*u_size, (self.mouse[0]+0.5)*u_size)   

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        
    def close(self):
        return None
        
        
if __name__ =="__main__":
    env = CatAndMouseEnv(8,8,mode_mouse=1)
    print("hello")
    env.reset()
    print(env.cat, env.mouse)
    env.render()
    x = input("press any key to exit")
    for _ in range(2000):
        env.render()
        a = env.action_space.sample()
        state, reward, isdone, info = env.step(a)
        print("{0}, {1}, {2}, {3}".format(a, reward, isdone, _))
        if isdone:
            input('input any key')
            break
        time.sleep(0.1)
    
    print("env closed")