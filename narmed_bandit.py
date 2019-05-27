# narmed_bandit.py
# Amir Harati may 2019
"""
    practicing n-armed bandit based on material in chap2. of Sutton Book.
"""
import numpy as np
import matplotlib.pyplot as plt


class Narmed_Bandit:
    def __init__(self, n):
        self.num_levers = n
        self.mean_rewards = np.random.random_integers(0, 1000, self.num_levers)
        self.var_rewards = np.random.random_integers(1, 1, self.num_levers) # allow for different lever to have different variance
        print("mean rewards: ", self.mean_rewards)

    def sample(self, lever):
        rewards = np.random.normal(self.mean_rewards, self.var_rewards)
        m = np.max(rewards)
        return {"reward": rewards[lever], "max_reward": m}



class Game:
    def __init__(self, num_actions, num_plays, eps, alpha, NB):
        self.num_actions =  num_actions
        self.num_plays = num_plays
        self.eps = eps
        self.alpha = alpha
        self.NB = NB

        self.Q = np.zeros([1, num_actions]) # values
        self.rewards = np.zeros([1, num_plays]) # it can also be negative for a play
        self.oracle_rewards = np.zeros([1, num_plays])


    def select_action(self,i):
        """ epsilon greedy"""
        greedy_action = np.argmax(self.Q[0,:])
        p = np.random.binomial(1, 1-np.min([0.999, 500.0*self.eps/i])) # some  heuristic to allow search more first
        
        if p == 1:
            a = greedy_action
        else:
            #print(i," ", p)
            a = np.random.randint(0,self.num_actions)
            #while(a == greedy_action):
            #    a = np.random.randint(0,self.num_actions)
            #print(a)
        x = self.NB.sample(a)
        r = x["reward"]
        mr = x["max_reward"]
        
        return a, r, mr

    def reset(self):
        self.Q = np.zeros([1, self.num_actions])
        self.rewards = np.zeros([1, self.num_plays]) # it can also be negative for a play
        self.oracle_rewards = np.zeros([1, self.num_plays])

    def update(self, a, r):
        self.Q[0, a] = self.Q[0, a] + self.alpha * (r - self.Q[0, a])

    def play(self):
        num_trails = 10
        avg_rewards = np.zeros([1, self.num_plays])
        avg_orc_rewards = np.zeros([1, self.num_plays])
        for j in range(num_trails ):
            self.reset()
            for i in range(self.num_plays):
                a, r, mr = self.select_action(i+1)
                self.rewards[0, i] += r
                self.oracle_rewards[0, i] += mr
                self.update(a, r)

            print("learned Q:", self.Q)
            avg_rewards += self.rewards
            avg_orc_rewards += self.oracle_rewards

        avg_rewards /= num_trails 
        avg_orc_rewards /= num_trails 

        #print(self.rewards)
        print("total rewards:", np.sum(avg_rewards))
        print("oracle total rewards:", np.sum(avg_orc_rewards))

        plt.plot(avg_rewards[0,:], "R")
        plt.plot(avg_orc_rewards[0,:], "B")
        plt.show()



if __name__ == "__main__":
    NB = Narmed_Bandit(10)

    game = Game(10, 10000, 1.0, 0.1, NB)
    game.play()

    #print(NB.sample(2))


        
    