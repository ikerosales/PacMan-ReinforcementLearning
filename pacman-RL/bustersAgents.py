from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import numpy as np
import os.path

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())
        
        
    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


class QLearningAgent(BustersAgent):

    # Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.epsilon = 0.1
        self.alpha = 0.0
        self.discount = 0.6
        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            # "*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(120)
        self.pacman_positions = [1, ]
        self.countActions = 0

    def initializeQtable(self, nrows):
        "Initialize qtable"
        self.q_table = np.zeros((nrows, len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def aux_func(self, state):
        a = [['North', 'South', 'East', 'West'], ['North', 'South', 'East'], ['North', 'South', 'West'],
             ['North', 'East', 'West'], ['South', 'East', 'West'], ['North', 'South'], ['East', 'West'],
             ['North', 'East'], ['North', 'West'], ['South', 'East'], ['South', 'West'], ['North'], ['South'],
             ['East'], ['West']]
        return a.index(state.getLegalPacmanActions()[:-1])

    def near_ghost(self, state):
        distances = state.data.ghostDistances
        min_d = 100
        for i in range(len(distances)):
            if distances[i] == None:
                distances[i] = -1
            elif distances[i] < min_d and distances[i] != -1:
                min_d = distances[i]
                near_index = i
        return near_index

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """
        near_index = self.near_ghost(state)
        (g_x, g_y) = state.getGhostPositions()[near_index]
        (p_x, p_y) = state.getPacmanPosition()
        diff_x = g_x - p_x
        diff_y = g_y - p_y

        # print("Distances" + str(distances))
        # print("Min_ dist" + str(min_d))
        # print("Near Index"+str(near_index))
        # print("gh_pos" + str(state.getGhostPositions()))
        # print(state.getLegalPacmanActions())

        # if diff_y >= diff_x  and diff_y > -diff_x:
        #     num = 0 # In north
        # elif diff_y < diff_x  and diff_y >= -diff_x:
        #     num = 1 # In east
        # elif diff_y <= diff_x and diff_y < -diff_x:
        #     num = 2 # In south
        # elif diff_y > diff_x  and diff_y <= -diff_x:
        #     num = 3 # In west
        # else :
        #     num = 0
        if diff_y > 0 and diff_x > 0:
            num = 0  # NE
        elif diff_y < 0 and diff_x > 0:
            num = 1  # SE
        elif diff_y < 0 and diff_x < 0:
            num = 2  # SW
        elif diff_y > 0 and diff_x < 0:
            num = 3  # NW
        elif diff_x == 0 and diff_y > 0:
            num = 4
        elif diff_x == 0 and diff_y < 0:
            num = 5
        elif diff_x > 0 and diff_y == 0:
            num = 6
        elif diff_x < 0 and diff_y == 0:
            num = 7
        return 8 * self.aux_func(state) + num

        # if diff_y > 0 and diff_x>0:
        #     return 4 #In line y = x cuadrant I
        # if diff_y < 0 and diff_x > 0:
        #     return 5 # In line y = -x cuadrant II
        # if diff_y < 0 and diff_x < 0:
        #     return 6 #In line y = x cuadrant III
        # if diff_y > 0 and diff_x < 0:
        #     return 7 #In line y = -x cuadrant IV
        # todo Hacer un for para crear una lista de todas las posibles combinaciones de paredes y luego tb con las 4 opciones

        # else :#diff_x == 0 and diff_y == 0:
        #     return 8
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):

        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]

    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions) == 0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
            return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            print("EPSILON CHOICE")
            return random.choice(legalActions)

        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        print(self.computePosition(state))
        if nextState.isWin():
            self.q_table[self.computePosition(state)][self.actions[action]] = \
                (1 - self.alpha) * self.getQValue(state, action) + \
                self.alpha * reward
        else:
            # print(self.computePosition(nextState))
            self.q_table[self.computePosition(state)][self.actions[action]] = \
                (1 - self.alpha) * self.getQValue(state, action) + \
                (self.alpha) * (reward + self.discount * self.computeValueFromQValues(nextState))

        "*** YOUR CODE HERE ***"

    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def distances_wo_nones(self, state):
        distances = state.data.ghostDistances
        for i in range(len(distances)):
            if distances[i] == None:
                distances[i] = -1
        return distances

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"
        rew = 0
        act_position = state.getPacmanPosition()
        next_position = nextstate.getPacmanPosition()
        near_gh_pos = state.getGhostPositions()[self.near_ghost(state)]
        self.pacman_positions.append(act_position)

        if self.distancer.getDistance(act_position, near_gh_pos) > self.distancer.getDistance(next_position,
                                                                                              near_gh_pos):
            rew += 5
        elif self.pacman_positions.count(next_position) >= 2:
            rew -= 3
        else:
            rew -= 5

        if sum(nextstate.getLivingGhosts()) - sum(state.getLivingGhosts()) != 0:
            self.pacman_positions = [1, ]
            # rew += 3
        # elif self.pacman_positions[-2] == next_position:
        #     rew -= 2

        if action not in state.getLegalPacmanActions():
            rew = -5
        print("reward: " + str(rew))
        return rew

        # if self.distancer.getDistance(act_position, near_gh_pos) > self.distancer.getDistance(next_position,
        #                                                                     near_gh_pos):
        #     rew += 1
        # elif self.pacman_positions.count(next_position) >= 3:
        #     rew -= 3
        # else:
        #     rew -= 1
        #
        # if sum(nextstate.getLivingGhosts()) - sum(state.getLivingGhosts()) != 0:
        #     self.pacman_positions = []
        #     # rew += 3
        #

        # rew = 0
        # self.countActions += 1
        # self.pacman_positions.append(state.getPacmanPosition()) #list of tuples
        # near_index = self.near_ghost(state)
        # distances = self.distances_wo_nones(state)
        # nextdistances = self.distances_wo_nones(nextstate)
        # if self.countActions > 50:
        #     self.pacman_positions = []
        #     self.countActions = 0
        #
        # if (self.computePosition(state) % 4) == 0 and action== "North" and "North" in state.getLegalPacmanActions():
        #     rew = 2
        # elif (self.computePosition(state)-1) % 4 == 0 and action=="East" and "East" in state.getLegalPacmanActions():
        #     rew = 2
        # elif (self.computePosition(state)-2) % 4 == 0 and action=="South" and "South" in state.getLegalPacmanActions():
        #     rew = 2
        # elif (self.computePosition(state)-3) % 4 == 0 and action == "West" and "West" in state.getLegalPacmanActions():
        #     rew = 2
        #
        # if sum(nextstate.getLivingGhosts())-sum(state.getLivingGhosts()) != 0:
        #     rew = 50
        #     self.pacman_positions = []
        #
        # if distances[near_index] < nextdistances[near_index] and \
        #    distances[near_index] != -1 and nextdistances[near_index] != -1:
        #     rew = -4
        # else:
        #     if (self.computePosition(state) % 4) == 0 and "North" not in state.getLegalPacmanActions():
        #         rew += 2
        #     elif (self.computePosition(
        #             state) - 1) % 4 == 0 and "East" not in state.getLegalPacmanActions():
        #         rew += 2
        #     elif (self.computePosition(
        #             state) - 2) % 4 == 0 and "South" not in state.getLegalPacmanActions():
        #         rew += 2
        #     elif (self.computePosition(
        #             state) - 3) % 4 == 0 and "West" not in state.getLegalPacmanActions():
        #         rew += 2
        #     else:
        #         rew = -4
        # if nextstate.getPacmanPosition() in self.pacman_positions:
        #     rew = -20
        # print("\n\nREWARD : " + str(rew)+"\n")
        # return rew

        "*** YOUR CODE HERE ***"

