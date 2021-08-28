import numpy as np
import pygame
import sys

from pygame.constants import K_0, K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_q, K_w, K_e, K_r, K_t, K_y, K_u, K_i, K_a, \
    K_s, K_d, K_f, K_g, K_h, K_j, K_k, K_z, K_x, K_c, K_v, K_b, K_n, K_m, K_COMMA, K_KP1, K_KP2, K_KP3, K_KP4, K_KP5, \
    K_KP6, K_KP7, K_KP8

from move_env.discrete5_move_env import Discrete5_Move_DQN_Env


save_path = 'dummy_expert_env'
env = Discrete5_Move_DQN_Env()

obs = env.reset()


pygame.init()  # 初始化pygame
size = width, height = 240, 360  # 设置窗口大小
screen = pygame.display.set_mode(size)  # 显示窗口
#ball = pygame.image.load('./img/mt.gif')  # 加载图片
ballrect=[]
for i in range(5):
    ballrect.append(pygame.Rect((0,0),(env.state[i,2],env.state[i,3])))  # 获取矩形区域
ground_rect = pygame.Rect((0,0),(45,120))
color = (30, 60, 0)  # 设置颜色
clock = pygame.time.Clock()  # 设置时钟
speed = [5, 5]  # 设置移动的X轴、Y轴
actions = []
observations = []
rewards = []
n_episodes = 1
episode_returns = np.zeros((n_episodes,))
episode_starts = []
while True:  # 死循环确保窗口一直显示
    clock.tick(60)  # 每秒执行60次
    for event in pygame.event.get():  # 遍历所有事件
        if event.type == pygame.QUIT:  # 如果单击关闭窗口，则退出

            numpy_dict = {
                'actions': actions,
                'obs': observations,
                'rewards': rewards,
                'episode_returns': episode_returns,
                'episode_starts': episode_starts
            }

            if save_path is not None:
                np.savez(save_path, **numpy_dict)
                print('episode_returns', episode_returns)
                print('reward_old',env.score_old)



            sys.exit()
        try:
            if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN):
                if event.key == K_1:
                    #ballrect[0] = ballrect[0].move([5,0])
                    action = 0

                    print(env.state[0, :])
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_2:
                    action = 1
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_3:
                    action = 2
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_4:
                    action = 3
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_5:
                    action = 4
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_6:
                    action = 5
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_7:
                    action = 6
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_8:
                    action = 7
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_q:
                    #ballrect[0] = ballrect[0].move([5,0])
                    action = 8
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_w:
                    action = 9
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_e:
                    action = 10
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_r:
                    action = 11
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_t:
                    action = 12
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_y:
                    action = 13
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_u:
                    action = 14
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_i:
                    action = 15
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_a:
                    # ballrect[0] = ballrect[0].move([5,0])
                    action = 16
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_s:
                    action = 17
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_d:
                    action = 18
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_f:
                    action = 19
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_g:
                    action = 20
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_h:
                    action = 21
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_j:
                    action = 22
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_k:
                    action = 23
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_z:
                    #ballrect[0] = ballrect[0].move([5,0])
                    action = 24
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_x:
                    action = 25
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_c:
                    action = 26
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_v:
                    action = 27
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_b:
                    action = 28
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_n:
                    action = 29
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_m:
                    action = 30
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_COMMA:
                    action = 31
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_KP1:
                    # ballrect[0] = ballrect[0].move([5,0])
                    action = 32
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_KP2:
                    action = 33
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward
                if event.key == K_KP3:
                    action = 34
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_KP4:
                    action = 35
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_KP5:
                    action = 36
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_KP6:
                    action = 37
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_KP7:
                    action = 38
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

                if event.key == K_KP8:
                    action = 39
                    actions.append(action)
                    observations.append(obs)

                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    episode_starts.append(done)
                    episode_returns[0] += reward

        except:
            pass
    #ball.fill([0,30,80])
    screen.fill(color)  # 填充颜色(设置为0，执不执行这行代码都一样)
    #screen.blit(ball, ballrect)  # 将图片画到窗口上
    for i in range(5):
        ballrect[i].update((env.state[i, 0] - env.state[i, 2] / 2, env.state[i, 1] - env.state[i, 3] / 2),
                           (env.state[i, 2], env.state[i, 3]))
    for i in range(5):
        pygame.draw.rect(screen,(0,(i+1)*20,0),ballrect[i])
    pygame.draw.rect(screen, (40, 20, 20), ground_rect,width=2)
    pygame.display.flip()  # 更新全部显示



