import random
import matplotlib as plt
import gym

amount_of_games = 10
amount_of_steps = 10

class QFunction:
    def __init__(self):
        self.weights = []
        self.discount_value = []
        self.replay_memory = []
        self.learning_rate = 0
        self.reward = 0
        self.rewards_memory = []
        self.old_q = 0

    def initializer(self, how_many_weights, discount_value, learning_rate, reward_memory, replay_memory):
        self.discount_value = discount_value
        self.learning_rate = learning_rate
        self.rewards_memory = reward_memory
        self.replay_memory = replay_memory

        for _ in range(how_many_weights):
            self.weights.append(random.uniform(0.1, how_many_weights))



    def q_function(self):

        q_value = 0
        for i in range(0,len(self.replay_memory),4):
            for j in range(len(self.weights)):
                weight_func = self.weights[j] * self.replay_memory[i+j]
                q_value = weight_func + q_value

        return q_value


    def reward_function(self, reward_gotten):
        'reward_funciton'
        q_reward = 0
        self.rewards_gotten.append(reward_gotten)
        q_reward = reward_gotten[1] + self.discount_value*self.q_function(self.replay_memory)-self.old_q

        return q_reward

    def weight_function(self):

        for i in range(len(self.weights)):
            self.weights[i] = self.weights + self.learning_rate*self.reward_function()*self.replay_memory[i]


        return self.weights

env = gym.make('CartPole-v0')
env.reset()
#obs = env.reset()

this_function = QFunction()





def initial_pop():
    replay_memory = []
    reward_memory = []

    for _ in range(amount_of_games):
        env.reset()
        for _ in range(amount_of_steps):
            observation, reward, done, info = env.step(env.action_space.sample())
            replay_memory.append(observation)
            reward_memory.append(reward)
            if done: break

    return replay_memory, reward_memory

replay_memory, reward_memory = initial_pop()

this_function.initializer(how_many_weights=4, discount_value=0.95, learning_rate=0.95, replay_memory=replay_memory, reward_memory=reward_memory)

print(this_function.q_function())
