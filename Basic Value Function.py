import random
import gym


amount_games = 1000
amount_of_steps = 200

gamma = 0.95
alpha = 0

t0, t1 = 5, 50


env = gym.make('CartPole-v0')
env.reset()

weight = []


def init(how_many_weights):
    for i in range(how_many_weights):
        weight.append(random.uniform(0.1, how_many_weights))
    #print('initial weights', weight)

def learning_schedule(t):
    return t0 /(t+t1)

def q_function(observation):
    #Q(s) = w1*f1(s) + w2*f2(s) + .. + wn*fn(s)
    q_value = 0

    for i in range(len(weight)):
        #Q(s) = wi*fi(s)
        q_value += weight[i] * f_function(observation)

    return q_value


def f_function(observation):
    f = 0
    for i in range(len(observation)):
       f += observation[i]

    return f

def delta_q(old_obs, new_obs, reward):

    delta_q_value = reward + gamma*(q_function(new_obs) - q_function(old_obs))

    return delta_q_value

def update_weights(old_obs, new_obs, reward):

    for i in range(len(weight)):
        weight[i] = weight[i] + alpha * delta_q(old_obs, new_obs, reward)*old_obs[i]
    #print('Updated weights', weight)

init(4)

overall_reward_final_games = 0
hit_target = False
for game in range(amount_games):
    overall_reward = 0
    observation = env.reset()
    for i in range(amount_of_steps):
        #env.render()
        q_value = q_function(observation=observation)
        #print('This is the Q Value', q_value)
        if q_value < 0:
            action = 0
        else:
            action = 1


        old_obseravtion = observation
        observation, reward, done, info = env.step(action)
        overall_reward += reward

        alpha = learning_schedule(game * amount_of_steps + i)
        if hit_target == False:
            update_weights(old_obs=old_obseravtion, new_obs=observation, reward=reward)

        if game > 500:
            overall_reward_final_games += reward

        if done: break


    #print(weight)
    #print(overall_reward)

print(overall_reward_final_games/500)
sum_of_weight = 0
for i in range(len(weight)):
    sum_of_weight  += weight[i]

print(sum_of_weight)