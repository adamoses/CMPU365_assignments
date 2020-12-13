# multiAgents.py
# --------------
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


import random

import util
from game import Agent, Directions
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [
            self.evaluationFunction(gameState, action)
            for action in legalMoves
        ]
        bestScore = max(scores)
        bestIndices = [
            index
            for index in range(len(scores))
            if scores[index] == bestScore
        ]
        chosenIndex = random.choice(
            bestIndices
        )  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates
        ]

        avg_scared_times = sum(newScaredTimes)/len(newScaredTimes)

        # get number of remaining food and manhattan distances to food pellets
        numFood = 0
        distances_to_food = []
        for x in list(range(0,newFood.width)):
            for y in list(range(0,newFood.height)):
                d = abs(newPos[0]-x)+abs(newPos[1]-y)
                if newFood[x][y]:
                    numFood += 1
                    distances_to_food.append(d)

        # compute average food manhattan distance 
        if len(distances_to_food) > 0:
            avg_distance_to_food = sum(distances_to_food)/len(distances_to_food)
        else: 
            avg_distance_to_food = 0

        # find manhattan distance to ghosts
        d_ghosts = []
        for ghost in newGhostStates:
            gx, gy = ghost.getPosition()
            d_ghosts.append(abs(newPos[0]-gx)+abs(newPos[1]-gy))

        avg_distance_to_ghost = 1

        if len(d_ghosts) > 0:
            avg_distance_to_ghost = (sum(d_ghosts)+1)/len(d_ghosts)
            
        return successorGameState.getScore() - avg_distance_to_food*0.2 - \
                numFood*2 + avg_scared_times*0.2 - 8/(avg_distance_to_ghost)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        numAgents = gameState.getNumAgents()

        utility, action = self.max_value(gameState, 1, numAgents)

        return action
        

    def max_value(self, gameState,curr_depth, numAgents):
        
        # Check for terminal state. Either win/loss or we reached depth limit
        if gameState.isWin() or gameState.isLose() or (curr_depth > self.depth):
            utility = self.evaluationFunction(gameState)
            return (utility, None)

        # initialize utility to none
        utility=None

        # get legal actions, and initialize variable to hold our chosen action
        legal_actions = gameState.getLegalActions()
        chosen_action = None

        for action in legal_actions:

            # create new game state resulted from taking an action
            newGameState = gameState.generateSuccessor(0, action)
            
            # create successor min nodes
            utility2, action2 = self.min_value(newGameState, curr_depth, 1, numAgents)
            
            # if this action has higher utility than current, take this action
            if utility == None or utility2 > utility:
                utility, chosen_action = (utility2, action)

        return (utility, chosen_action)

    def min_value(self, gameState, curr_depth, agent, numAgents):

        # Check for terminal state. Either win/loss
        if gameState.isWin() or gameState.isLose():
            utility = self.evaluationFunction(gameState)
            return (utility, None)

        # initialize utility to none
        utility=None

        # get legal actions, and initialize variable to hold our chosen action
        legal_actions = gameState.getLegalActions(agent)
        chosen_action = None 

        for action in legal_actions:

            # create new game state resulted from taking an action
            newGameState = gameState.generateSuccessor(agent, action)

            # create successor nodes (either max nodes if we are on the last agent, or min nodes for next agent)
            if agent == numAgents - 1:
                utility2, action2 = self.max_value(newGameState, curr_depth+1, numAgents)
            else:
                utility2, action2 = self.min_value(newGameState, curr_depth, agent+1, numAgents)
            
            # if this action has lower utility than current, take this action
            if utility == None or utility2 < utility:
                utility, chosen_action = (utility2, action)

        return (utility, chosen_action)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        utility, action = self.max_value(gameState, 1, numAgents, None, None)

        return action

    def max_value(self, gameState,curr_depth, numAgents, alpha, beta):
        
        # Check for terminal state. Either win/loss or we reached depth limit
        if gameState.isWin() or gameState.isLose() or (curr_depth > self.depth):
            utility = self.evaluationFunction(gameState)
            return (utility, None)

        # initialize utility to none
        utility=None

        # get legal actions, and initialize variable to hold our chosen action
        legal_actions = gameState.getLegalActions()
        chosen_action = None

        for action in legal_actions:

            # create new game state resulted from taking an action
            newGameState = gameState.generateSuccessor(0, action)
            
            # create successor min nodes
            utility2, action2 = self.min_value(newGameState, curr_depth, 1, numAgents, alpha, beta)
            
            # if this action has higher utility than current, take this action
            if utility == None or utility2 > utility:
                utility, chosen_action = (utility2, action)
                if alpha == None:
                    alpha = utility
                else:
                    alpha = max(alpha, utility)
            if not beta == None and not utility == None and utility > beta:
                return (utility, chosen_action)

        return (utility, chosen_action)

    def min_value(self, gameState, curr_depth, agent, numAgents, alpha, beta):

        # Check for terminal state. Either win/loss
        if gameState.isWin() or gameState.isLose():
            utility = self.evaluationFunction(gameState)
            return (utility, None)

        # initialize utility to none
        utility=None

        # get legal actions, and initialize variable to hold our chosen action
        legal_actions = gameState.getLegalActions(agent)
        chosen_action = None 

        for action in legal_actions:

            # create new game state resulted from taking an action
            newGameState = gameState.generateSuccessor(agent, action)

            # create successor nodes (either max nodes if we are on the last agent, or min nodes for next agent)
            if agent == numAgents - 1:
                utility2, action2 = self.max_value(newGameState, curr_depth+1, numAgents, alpha, beta)
            else:
                utility2, action2 = self.min_value(newGameState, curr_depth, agent+1, numAgents, alpha, beta)
            
            # if this action has lower utility than current, take this action
            if utility == None or utility2 < utility:
                utility, chosen_action = (utility2, action)
                if beta == None:
                    beta = utility
                else:
                    beta = min(beta, utility)
            if not alpha == None and not utility == None and utility < alpha:
                return (utility, chosen_action)

        return (utility, chosen_action)



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        utility, action = self.pacman_turn(gameState, 1, numAgents)

        return action

    def pacman_turn(self, gameState, curr_depth, numAgents):

        # Check for terminal state. Either win/loss or we reached depth limit
        if gameState.isWin() or gameState.isLose() or (curr_depth > self.depth):
            utility = self.evaluationFunction(gameState)
            return (utility, None)

        # get possible moves
        possible_moves = gameState.getLegalActions()

        # initiliaze variables for chosen move and its utility
        chosen_move = None
        max_util = None

        for move in possible_moves:

            # generate successor state and node (pacman's successor is always ghost)
            newGameState = gameState.generateSuccessor(0, move)
            new_utility, newMove = self.ghost_turn(newGameState, curr_depth, 1, numAgents)
            
            # choose maximum utility
            if chosen_move == None or new_utility > max_util:
                chosen_move = move
                max_util = new_utility
        
        return (max_util, chosen_move)

    def ghost_turn(self, gameState, curr_depth, agent, numAgents):

        # Check for terminal state. Either win/loss
        if gameState.isWin() or gameState.isLose():
            utility = self.evaluationFunction(gameState)
            return (utility, None)

        # get legal actions, initilize empty list for all possible utilities
        possible_moves = gameState.getLegalActions(agent)
        utils = []

        for move in possible_moves:

            utility = None
            # Generate successor state given an action
            newGameState = gameState.generateSuccessor(agent, move)

            # create successor nodes, pacman if out of agents, the next agent otherwise
            if agent == numAgents - 1:
                utility, action = self.pacman_turn(newGameState, curr_depth+1, numAgents)
            else:
                utility, action = self.ghost_turn(newGameState, curr_depth, agent+1, numAgents)

            # append new utility
            utils.append(utility)
            
        # average utilities uniformly 
        avg_utils = sum(utils)/len(utils)
        
        return (avg_utils, None)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    gameState = currentGameState
    position = gameState.getPacmanPosition()
    food = gameState.getFood()
    ghostStates = gameState.getGhostStates()
    scaredTimes = [
        ghostState.scaredTimer for ghostState in ghostStates
    ]

    avg_scared_times = sum(scaredTimes)/len(scaredTimes)

    # get number of remaining food and manhattan distances to food pellets
    numFood = 0
    distances_to_food = []
    for x in list(range(0,food.width)):
        for y in list(range(0,food.height)):
            d = abs(position[0]-x)+abs(position[1]-y)
            if food[x][y]:
                numFood += 1
                distances_to_food.append(d)

    # compute average food manhattan distance 
    if len(distances_to_food) > 0:
        avg_distance_to_food = sum(distances_to_food)/len(distances_to_food)
    else: 
        avg_distance_to_food = 0

    # find manhattan distance to ghosts
    d_ghosts = []
    for ghost in ghostStates:
        gx, gy = ghost.getPosition()
        d_ghosts.append(abs(position[0]-gx)+abs(position[1]-gy))

    avg_distance_to_ghost = 1

    if len(d_ghosts) > 0:
        avg_distance_to_ghost = (sum(d_ghosts)+1)/len(d_ghosts)
        
    return gameState.getScore() - avg_distance_to_food*0.2 - \
            numFood*2 + avg_scared_times*0.2 - 8/(avg_distance_to_ghost)


# Abbreviation
better = betterEvaluationFunction
