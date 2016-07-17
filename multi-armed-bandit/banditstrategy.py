import scipy.stats as stats
import numpy as np
import random


class BanditStrategy(object):
    '''
    Implements an online, learning strategy to solve
    the Multi-Armed Bandit problem.

    parameters:
        bandits: a Bandit class with .pull method
		choice_function: accepts a self argument (which gives access to all the variables), and
						returns and int between 0 and n-1
    methods:
        sample_bandits(n): sample and train on n pulls.

    attributes:
        N: the cumulative number of samples
        choices: the historical choices as a (N,) array
        bb_score: the historical score as a (N,) array
    '''

    def __init__(self, bandits, choice_function):
        '''
        INPUT: Bandits, function

        Initializes the BanditStrategy given an instance of the Bandits class
        and a choice function.
        '''
        self.bandits = bandits
        n_bandits = len(self.bandits)
        self.wins = np.zeros(n_bandits)
        self.trials = np.zeros(n_bandits)
        self.N = 0
        self.choices = []
        self.score = []
        self.choice_function = choice_function

    def sample_bandits(self, n=1):
        '''
        INPUT: int
        OUTPUT: None

        Simulate n rounds of running the bandit machine.
        '''
        score = np.zeros(n)
        choices = np.zeros(n)

        # seed the random number generators so you get the same results every
        # time.
        #np.random.seed(101)
        #random.seed(101)

        for k in range(n):
            #sample from the bandits's priors, and select the largest sample
            choice = self.choice_function(self)

            #sample the chosen bandit
            result = self.bandits.pull(choice)

            #update priors and score
            self.wins[choice] += result
            self.trials[choice] += 1
            score[k] = result
            self.N += 1
            choices[k] = choice

        self.score = np.r_[self.score, score]
        self.choices = np.r_[self.choices, choices]


def max_mean(self):
    '''
    Pick the bandit with the current best observed proportion of winning.
    Return the index of the winning bandit.
    '''
    # make sure to play each bandit at least once
    if len(self.trials.nonzero()[0]) < len(self.bandits):
        return np.random.randint(0, len(self.bandits))
    return np.argmax(self.wins / (self.trials + 1))

def random_choice(self):
    '''
    Pick a bandit uniformly at random.
    Return the index of the winning bandit.
    '''
    return np.random.randint(0, len(self.wins))

def epsilon_greedy(self, epsilon=0.1):
    '''
    Pick a bandit uniformly at random epsilon percent of the time.
    Otherwise pick the bandit with the best observed proportion of winning.
    Return the index of the winning bandit.
    '''
    if stats.bernoulli(epsilon).rvs() == 1:
        return random_choice(self)
    else:
        return max_mean(self)

def softmax(self, tau=0.001):
    '''
    Pick an bandit according to the Boltzman Distribution.
    Return the index of the winning bandit.
    '''
    win_prop = np.array([0 if self.trials[i] == 0
                         else float(self.wins[i])/self.trials[i]
                         for i in range(len(self.bandits))])
    boltz_nums = np.exp(win_prop * tau)
    boltz_probs = boltz_nums/sum(boltz_nums)
    draw = np.random.multinomial(1,boltz_probs)
    return list(draw).index(max(draw))


def ucb1(self):
    '''
    Pick the bandit according to the UCB1 strategy.
    Return the index of the winning bandit.
    '''
    win_prop = np.array([0 if self.trials[i] == 0
                         else float(self.wins[i])/self.trials[i]
                         for i in range(len(self.bandits))])
    UCB = win_prop + (2 * np.log(sum(self.trials))/self.trials)
    return list(UCB).index(max(UCB))

def bayesian_bandit(self):
    '''
    Randomly sample from a beta distribution for each bandit and pick the one
    with the largest value.
    Return the index of the winning bandit.
    '''
    bandit_draws = []
    for i in range(len(self.bandits)):
        bandit_draws.append( stats.beta(1+self.wins[i],1+self.trials[i]-self.wins[i]).rvs())
    return bandit_draws.index(max(bandit_draws))
