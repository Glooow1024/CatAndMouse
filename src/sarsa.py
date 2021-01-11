#!/bin/env/python
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as c
from CatEnv import CatAndMouseEnv


class SARSA:

    def __init__(self, height=4,width=4,mode_obstacle=0,mode_mouse=0,map=None,Q=None,mouse=None):
        self.mode_obstacle = mode_obstacle
        self.mode_mouse = mode_mouse

        if map is not None:
            self.height,self.width = map.shape
            self.num_states,self.num_action = Q.shape
            self.env = CatAndMouseEnv(mode_obstacle=mode_obstacle,mode_mouse=mode_mouse,map=map,mouse=mouse)
        else:
            self.height = height
            self.width = width
            if self.mode_mouse:
                self.num_states = self.height * self.width * self.height * self.width
            else:
                self.num_states = self.height * self.width
            self.num_action = 4
            self.env = CatAndMouseEnv(self.height,self.width,self.mode_obstacle,self.mode_mouse)
        
        if Q is not None:
            self.Q = Q
        else:
            self.Q = np.zeros((self.num_states,self.num_action))
        
        self.alpha = 0.1
        self.gamma = 1.0
        self.epsilon = 0.1
        
        self.max_count = 1000
        self.Epoch = 1000
        if self.mode_mouse:
            self.max_count *= 10
            self.Epoch *= 10
        if self.mode_obstacle:
            self.Epoch *= 10


    def learn(self):
        self.rewards = []
        for e in range(self.Epoch):
            # initialize
            cat_mouse_pos = self.env.reset()
            state_cur = self.pos2state(cat_mouse_pos)
            done = False
            count = 0
            total_reward = 0.0

            action, _ = self.epsilon_greedy(cat_mouse_pos)

            while (not done) and (count < self.max_count):
                # step forward
                cat_mouse_pos, r, done, _ = self.env.step(action)
                state_new = self.pos2state(cat_mouse_pos)
                action_new, _ = self.epsilon_greedy(cat_mouse_pos)

                # update Q
                if done:
                    self.Q[state_cur][action] = self.Q[state_cur][action] + self.alpha * \
                        (r + self.gamma * 0.0 - self.Q[state_cur][action])
                else:
                    self.Q[state_cur][action] = self.Q[state_cur][action] + self.alpha * \
                        (r + self.gamma*self.Q[state_new][action_new] - self.Q[state_cur][action])

                # update state
                total_reward += r
                state_cur = state_new
                action = action_new
                count += 1

            self.rewards.append(total_reward)
            if e%100 == 99:
                print('SARSA Epoch ', e)

        print(self.Q)


    def prediction(self, cat_mouse_pos):
        state_idx = self.pos2state(cat_mouse_pos)
        a = np.argmax(self.Q[state_idx])
        return a


    def test(self):
        # initialize
        cat_mouse_pos = self.env.reset()
        reward = 0
        done = False
        count = 0
        r = 0
        self.env.render()

        while (not done) and (count < self.max_count):
            action_map = ['Down', 'Right', 'Up', 'Left']
            # step forward
            action = self.prediction(cat_mouse_pos)
            print('Cat:',tuple(cat_mouse_pos[0]), ' Mouse:',tuple(cat_mouse_pos[1]), ' Action:',action_map[action], \
                ' Step reward:',r, ' Total reward:',reward, ' Count: ',count)
            cat_mouse_pos, r, done, _ = self.env.step(action)
            reward += r
            
            # update state
            self.env.render()
            count += 1
            time.sleep(0.1)

            if done:
                input('Cat chased mouse! Input to stop...')

        print('Result: ', done, reward)


    def pos2state(self, cat_mouse_pos):
        cat = cat_mouse_pos[0]
        mouse = cat_mouse_pos[1]
        if self.mode_mouse:
            state_idx = (cat[0]*self.width+cat[1])*(self.height*self.width) + mouse[0]*self.width+mouse[1]
        else:
            state_idx = cat[0]*self.width+cat[1]
        return state_idx


    def epsilon_greedy(self, cat_mouse_pos):
        
        state_idx = self.pos2state(cat_mouse_pos)
        action = np.argmax(self.Q[state_idx])
        prob = np.ones(len(self.env.action_map)) * self.epsilon / len(self.env.action_map)
        prob[action] += 1-self.epsilon

        action = np.random.choice(range(len(self.env.action_map)), p=prob)
        
        return action, prob


    def visualization(self):
        # https://matplotlib.org/examples/color/colormaps_reference.html
        map = self.env.world
        #map[self.env.mouse[0]][self.env.mouse[1]] = 2   # mouse
        cMap = plt.cm.get_cmap('PuBuGn')#('GnBu')#('YlOrRd')
        plt.pcolormesh(map, cmap=cMap, edgecolor=(0,0,0),linewidths=1,linestyles='dashed')
        #plt.axis(xlim=[0,self.height],ylim=[0,self.width])
        plt.axis('equal')
        if self.mode_mouse:
            plt_mouse = np.array([6,1])
        else:
            plt_mouse = self.env.mouse
        plt.axis('equal')
        for x in range(self.width):
            for y in range(self.height):
                state_idx = self.pos2state(((y,x),plt_mouse))
                action = np.argmax(self.Q[state_idx])
                action = self.env.action_map[action]
                if x==plt_mouse[1] and y==plt_mouse[0]:
                    plt.plot(x+0.5,y+0.5,'*',markersize=100/np.sqrt(self.height))
                elif not self.env.world[y][x]:
                    plt.quiver(x+0.5-action[1]*0.25,y+0.5-action[0]*0.25,action[1],action[0],(0,0,1), width=0.04/self.height,scale=2,scale_units='y')
        plt.title('SARSA')
        plt.show()




if __name__ == "__main__":
    mode_obstacle = 0
    mode_mouse = 0
    q = SARSA(8,8,mode_obstacle,mode_mouse)
    q.learn()
    np.savez('sarsa_%d_%d.npz'%(mode_obstacle,mode_mouse), map=q.env.world, \
        mouse=q.env.mouse, Q=q.Q, mode_obstacle=mode_obstacle, mode_mouse=mode_mouse)
    print('1. Learn done!')

    mode_obstacle = 0
    mode_mouse = 1
    q = SARSA(8,8,mode_obstacle,mode_mouse)
    q.learn()
    np.savez('sarsa_%d_%d.npz'%(mode_obstacle,mode_mouse), map=q.env.world, \
        mouse=q.env.mouse, Q=q.Q, mode_obstacle=mode_obstacle, mode_mouse=mode_mouse)
    print('2. Learn done!')

    mode_obstacle = 1
    mode_mouse = 1
    q = SARSA(8,8,mode_obstacle,mode_mouse)
    q.learn()
    np.savez('sarsa_%d_%d.npz'%(mode_obstacle,mode_mouse), map=q.env.world, \
        mouse=q.env.mouse, Q=q.Q, mode_obstacle=mode_obstacle, mode_mouse=mode_mouse)
    print('3. Learn done!')
    '''q.visualization()
    input('Input to start test...')
    q.test()
    print('Test done!')'''
