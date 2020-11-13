# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import collections

import mdp
import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration() 

    def runValueIteration(self):
        # Write value iteration code here
        # *** YOUR CODE HERE ***

        while self.iterations > 0:
            
            # doing batch iteration, don't want to update values until after all the states are updated
            newValues = util.Counter()

            # for each state
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):

                    # set new value of state to be the max Q value for each possible action
                    newValues[state] = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])

            self.iterations -= 1
            self.values = newValues

        return self.values

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        # *** YOUR CODE HERE ***'

        # get possible new states and their respective probabilities
        possibleStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)

        # compute q value by summing up over each possible new state
        QValues = [(self.mdp.getReward(state, action, newState) + self.discount*self.values[newState])*prob for newState, prob in possibleStatesAndProbs]
        return sum(QValues)
       

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit. Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        # *** YOUR CODE HERE ***
        # get possible actions and compute their q values
        actions = self.mdp.getPossibleActions(state)
        QValues = [(action, self.computeQValueFromValues(state, action)) for action in actions]

        # initialize variable to hold max possible q value from actions
        maxAction = (None, None)

        for i in range(len(actions)):

            if maxAction[0] == None or QValues[i][1] > maxAction[1]:
                maxAction = QValues[i]

        return maxAction[0]


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
