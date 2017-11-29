import random
import gym
import numpy as np

amount_games = 5000
amount_of_steps = 200

gamma = 0.95
alpha = 0

t0, t1 = 5, 50


env = gym.make('CartPole-v0')
env.reset()

weight = []


def init(how_many_weights):
    for i in range(how_many_weights):
        weight.append(0)
    #print('initial weights', weight)

def learning_schedule(t):
    return t0 /(t+t1)

def q_function(observation, action):
    #Q(s) = w1*f1(s) + w2*f2(s) + .. + wn*fn(s)
    q_value = 0

    for i in range(len(weight)):

        #Q(s) = wi*fi(s)
        #q_value += weight[i] * f_function(observation[i], action)
        q_value += weight[i] * (observation[i] * action)

    return q_value


def f_function(observation, action):
    #f = 0

    #for i in range(len(observation)):
    f = observation + action

    return f

def delta_q(old_obs, new_obs, reward, action):
    if action == 1:
        other_action = -1
    else: other_action = 1
    delta_q_value = reward + gamma*np.argmax([q_function(new_obs,action), q_function(new_obs,other_action)]) - q_function(old_obs,action)

    return delta_q_value

def update_weights(old_obs, new_obs, reward, action):

    for i in range(len(weight)):
        weight[i] = weight[i] + alpha * delta_q(old_obs, new_obs, reward, action)*(old_obs[i] * action)
    #print('Updated weights', weight)

init(4)

overall_reward_final_games = 0
hit_target = False
for game in range(amount_games):
    overall_reward = 0
    observation = env.reset()
    for i in range(amount_of_steps):
        #if game > 950:
        #env.render()
        if game == 0:
            action = 0
            q_action = -1

        q_value = q_function(observation=observation, action=-1)
        q_value1 = q_function(observation=observation, action=1)
        #print('This is the Q Value', q_value)
        if q_value > q_value1:
            action = 0
        else:
            action = 1


        old_obseravtion = observation
        observation, reward, done, info = env.step(action)
        overall_reward += reward

        alpha = learning_schedule(game * amount_of_steps + i)
        if hit_target == False:
            update_weights(old_obs=old_obseravtion, new_obs=observation, reward=reward, action=action)

        if game > (amount_games - 500):
            overall_reward_final_games += reward

        if done: break


    #print(weight)
    #print(overall_reward)

print(overall_reward_final_games/500)
sum_of_weight = 0
for i in range(len(weight)):
    sum_of_weight  += weight[i]

print(sum_of_weight)