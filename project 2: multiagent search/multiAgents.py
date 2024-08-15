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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        minFoodDistance = 9999999
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            if distance < minFoodDistance:
                minFoodDistance = distance

        if minFoodDistance != 9999999 and minFoodDistance > 0:
            score+=1 / minFoodDistance


        minGhostDistance = 9999999
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0:
                distance = manhattanDistance(newPos,ghostState.getPosition())
                if distance<minGhostDistance:
                    minGhostDistance = distance

        if minGhostDistance != 9999999 and minGhostDistance>0:
            score -= 2 / minGhostDistance


        scaredGhostCount=0
        for ghostState in newGhostStates:
            if ghostState.scaredTimer>0:
                scaredGhostCount+=1

        score+= scaredGhostCount*5


        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        def minimax(state, depth, agentIndex):

            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                return maxValue(state, depth)
            else:
                return minValue(state, depth, agentIndex)

        def maxValue(state, depth):
            choice = -9999999
            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor(0, action)
                nextAgent = 1
                choice = max(choice, minimax(nextState, depth, nextAgent))
            return choice

        def minValue(state, depth, agentIndex):
            choice = 9999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                choice = min(choice, minimax(nextState, nextDepth, nextAgent))
            return choice

        bestScore = -9999999
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(successor, 0, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaSearch(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:
                return maxValue(state, depth, agentIndex, alpha, beta)
            else:
                return minValue(state, depth, agentIndex, alpha, beta)

        def maxValue(state, depth, agentIndex, alpha, beta):
            choice = -9999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(0, action)
                nextAgent = 1
                choice = max(choice, alphaBetaSearch(nextState, depth, nextAgent, alpha, beta))
                if choice > beta:
                    return choice
                alpha = max(alpha, choice)
            return choice


        def minValue(state, depth, agentIndex, alpha, beta):
            choice = 9999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                if nextAgent == 0:
                    nextDepth = depth + 1
                else:
                    nextDepth = depth
                choice = min(choice, alphaBetaSearch(nextState, nextDepth, nextAgent, alpha, beta))
                if choice < alpha:
                    return choice
                beta = min(beta, choice)
            return choice



        alpha = -9999999
        beta = 9999999
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            value = alphaBetaSearch(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if value > alpha:
                alpha = value
                bestAction = action
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state, depth, agentIndex):

            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state)

            if agentIndex == 0:
                return maxValue(state, depth, agentIndex)
            else:
                return expValue(state, depth, agentIndex)

        def maxValue(state, depth, agentIndex):
            choice = -9999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                choice = max(choice, expectimax(nextState, depth+1, nextAgent))
            return choice


        def expValue(state, depth, agentIndex):
            choice = 0
            actions = state.getLegalActions(agentIndex)
            if actions:
                probability = 1.0 / len(actions)
            else:
                probability = 1.0

            for action in actions:
                nextState = state.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                choice += probability * expectimax(nextState, depth + 1, nextAgent)
            return choice



        bestScore = -9999999
        bestAction = Directions.STOP
        for action in gameState.getLegalActions(0):
            score = expectimax(gameState.generateSuccessor(0, action), 1, 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <it's written based on the distance to the nearest food,
    the distance to ghosts, and the number of remaining food pellets.
    Closer food is better, less remaining food is better and more impact of ghost is better>
    """

    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    score = currentGameState.getScore()
    foodList = newFood.asList()
    minFoodDistance = min(manhattanDistance(newPos, food) for food in foodList) if foodList else 0


    ghostImpact = 0
    for ghost in newGhostStates:
        ghostDistance = manhattanDistance(ghost.getPosition(), newPos)
        if ghost.scaredTimer == 0:
            if ghostDistance < 3:
                ghostImpact -= (3 - ghostDistance) * 10

        else:
            ghostImpact += 8 if ghostDistance < 3 else 0

    remainingFood = len(foodList)

    score += ghostImpact
    score -= 2 * minFoodDistance
    score -= 4 * remainingFood


    return score


# Abbreviation
better = betterEvaluationFunction

