# myTeam.py
# ---------------
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


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from ast import Raise
from typing import List, Tuple

import numpy as np

from numpy import true_divide
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys, os
from capture import GameState, noisyDistance
from game import Directions, Actions, AgentState, Agent
from util import nearestPoint
import sys,os

# the folder of current file.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

from lib_piglet.utils.pddl_solver import pddl_solver
from lib_piglet.domains.pddl import pddl_state
from lib_piglet.utils.pddl_parser import Action


# from lib_piglet.expanders import grid_expander
# from lib_piglet.domains import gridmap
# from lib_piglet.search.graph_search import graph_search
from lib_piglet.utils.data_structure import bin_heap
from lib_piglet.search.search_node import compare_node_f, search_node

CLOSE_DISTANCE = 4
MEDIUM_DISTANCE = 15
LONG_DISTANCE = 25


#################
# Team creation #
#################

def manhattan_heuristic(current_state, goal_state):
    return abs(current_state[0] - goal_state[0]) + abs(current_state[1] - goal_state[1])

class grid_action:

    def __init__(self, move, cost):
        self.move_ = move
        self.cost_ = cost


def createTeam(firstIndex, secondIndex, isRed,
                             first = 'MixedAgent', second = 'MixedAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########                                       

class MixedAgent(CaptureAgent):
    """
    This is an agent that use pddl to guide the high level actions of Pacman
    """
    # Default weights for q learning, if no QLWeights.txt find, we use the following weights.
    # You should add your weights for new low level planner here as well.
    # weights are defined as class attribute here, so taht agents share same weights.
    QLWeights = {'offensiveWeights': {'teamDistance': 10.038047157608233, 'closest-capsule': -6.549717747427919, 'closest-food': -12.397551732563246, 'bias': -482.6007490020454, '#-of-ghosts-1-step-away': -5.430154426151799, 'successorScore': 5.7223547717226495, 'chance-return-food': 1.3654965716254632}, 'defensiveWeights': {'scared-ghost': -5000, 'centre_dist': 25, 'numInvaders': -1000.0, 'onDefense': 100, 'teamDistance': 10, 'invaderDistance': -10, 'stop': -14.134607644235528, 'reverse': -7.027542855355465}, 'escapeWeights': {'onDefense': 1000, 'enemyDistance': 30, 'stop': -100, 'distanceToHome': -20}}
    QLWeightsFile = BASE_FOLDER+'/QLWeightsMyTeam.txt'

    # Also can use class variable to exchange information between agents.
    CURRENT_ACTION = {}
    TARGET_LOCATION = {}

    def registerInitialState(self, gameState: GameState):
        self.pddl_solver = pddl_solver(BASE_FOLDER+'/myTeam.pddl')
        self.highLevelPlan: List[Tuple[Action,pddl_state]] = None # Plan is a list Action and pddl_state
        self.currentNegativeGoalStates = []
        self.currentPositiveGoalStates = []
        self.currentActionIndex = 0 # index of action in self.highLevelPlan should be execute next

        self.startPosition = gameState.getAgentPosition(self.index) # the start location of the agent
        CaptureAgent.registerInitialState(self, gameState)

        self.lowLevelPlan: List[Tuple[str,Tuple]] = []
        self.lowLevelActionIndex = 0

        self.repeating = False

        # REMEMBER TRUN TRAINNING TO FALSE when submit to contest server.
        self.trainning = False # trainning mode to true will keep update weights and generate random movements by prob.
        self.epsilon = 0.2 #default exploration prob, chance to take a random step
        self.alpha = 0.035 #default learning rate
        self.discountRate = 0.9 # default discount rate on successor state q value when update
        
        # Use a dictionary to save information about current agent.
        MixedAgent.CURRENT_ACTION[self.index]={}

        MixedAgent.TARGET_LOCATION[self.index] = None

        # calculate bottlenecks at beginning of the game
        # initialised as empty list as we do not use bottlenecks due to server crashing
        # reverts to food densities in the defensive agent
        self.bottlenecks = []

        # track last disappeared food to always move towards it if it exists in defence mode
        # resets if invader is found or plan changes
        self.disappeared_food_lock = None
        """
        Open weights file if it exists, otherwise start with empty weights.
        NEEDS TO BE CHANGED BEFORE SUBMISSION

        """
        # if os.path.exists(MixedAgent.QLWeightsFile):
        #     with open(MixedAgent.QLWeightsFile, "r") as file:
        #         MixedAgent.QLWeights = eval(file.read())
        #     print("Load QLWeights:",MixedAgent.QLWeights )
        
    
    def final(self, gameState : GameState):
        """
        This function write weights into files after the game is over. 
        You may want to comment (disallow) this function when submit to contest server.
        """
        # print("Write QLWeights:", MixedAgent.QLWeights)
        # file = open(MixedAgent.QLWeightsFile, 'w')
        # file.write(str(MixedAgent.QLWeights))
        # file.close()
    

    def chooseAction(self, gameState: GameState):
        """
        This is the action entry point for the agent.
        In the game, this function is called when its current agent's turn to move.

        We first pick a high-level action.
        Then generate low-level action ("North", "South", "East", "West", "Stop") to achieve the high-level action.
        """

        #-------------High Level Plan Section-------------------
        # Get high level action from a pddl plan.
        
        # Collect objects and init states from gameState
        objects, initState = self.get_pddl_state(gameState)
        positiveGoal, negtiveGoal = self.getGoals(objects,initState)

        # print(self.highLevelPlan)
        # print(self.currentActionIndex)
        # Check if we can stick to current plan 
        if not self.stateSatisfyCurrentPlan(initState, positiveGoal, negtiveGoal):
            # Cannot stick to current plan, prepare goals and replan
            print("Agent:",self.index,"compute plan:")
            print("\tOBJ:"+str(objects),"\tINIT:"+str(initState), "\tPOSITIVE_GOAL:"+str(positiveGoal), "\tNEGTIVE_GOAL:"+str(negtiveGoal),sep="\n")
            self.highLevelPlan: List[Tuple[Action,pddl_state]] = self.getHighLevelPlan(objects, initState,positiveGoal, negtiveGoal) # Plan is a list Action and pddl_state
            self.currentActionIndex = 0
            self.lowLevelPlan = [] # reset low level plan

            self.currentPositiveGoalStates = positiveGoal
            self.currentNegativeGoalStates = negtiveGoal

            print("\tPLAN:",self.highLevelPlan)
            
        # if a high level plan isnt returned, we will default to defence 
        if len(self.highLevelPlan)==0:

            raise Exception("Solver returned empty plan, you need to think how you handle this situation or how you modify your model ")
        
        # 
        highLevelAction = self.highLevelPlan[self.currentActionIndex][0].name

        # replan set to false unless we necessitate it
        replan = False
        MixedAgent.CURRENT_ACTION[self.index] = highLevelAction
        print("Agent:", self.index, highLevelAction)

        # pings food has disappeared, we need to replan and handle in low level solver.
        if len(self.getFoodDisappeared(gameState)) > 0:
            replan = True

        # agent in sight, replan
        # enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        # for enemy_index, enemy_state in enemies:
        #     enemy_position = enemy_state.getPosition()

        #     if enemy_position != None:
        #         if self.getMazeDistance(gameState.getAgentPosition(self.index), enemy_position) <= CLOSE_DISTANCE:
        #             replan = True


        #-------------Low Level Plan Section-------------------
        # Get the low level plan using Q learning, and return a low level action at last.
        # A low level action is defined in Directions, whihc include {"North", "South", "East", "West", "Stop"}
        
        # if current agent is pacman and not powered up, if ghost in the way replan.
        # op_locs = self.getGhostLocs(gameState)
        # check = gameState.getAgentState(self.index).isPacman and any(item in op_locs for item in self.lowLevelPlan)
   
        # check if agent is repeating actions.
        if self.isAgentRepeatingActions():
            #print("repeating actions...")
            replan = True
            self.repeating = True

        # if low level plan unsatisifed or need to replan
        if not self.posSatisfyLowLevelPlan(gameState) or replan:
            
            # plan agent using low level heuristic search
            self.lowLevelPlan = self.getLowLevelPlanHS(gameState, highLevelAction) #Generate low level plan with heuristic search

            # if no plan for agent using HS, fall back on QL
            if len(self.lowLevelPlan) == 0:
            # else:
                print(f'no plan for {highLevelAction}, default to QL')
                self.lowLevelPlan = self.getLowLevelPlanQL(gameState, highLevelAction) #Generate low level plan with q learning
            
            self.lowLevelActionIndex = 0

        lowLevelAction = self.lowLevelPlan[self.lowLevelActionIndex][0]
        lowLevelPos = self.lowLevelPlan[self.lowLevelActionIndex][1]

        self.lowLevelActionIndex+=1
        print("\tAgent:", self.index, lowLevelAction, lowLevelPos)
        return lowLevelAction

    #------------------------------- PDDL and High-Level Action Functions ------------------------------- 
    
    
    def getHighLevelPlan(self, objects, initState, positiveGoal, negtiveGoal) -> List[Tuple[Action,pddl_state]]:
        """
        This function prepare the pddl problem, solve it and return pddl plan
        """
        # Prepare pddl problem
        self.pddl_solver.parser_.reset_problem()
        self.pddl_solver.parser_.set_objects(objects)
        self.pddl_solver.parser_.set_state(initState)
        self.pddl_solver.parser_.set_negative_goals(negtiveGoal)
        self.pddl_solver.parser_.set_positive_goals(positiveGoal)
        
        # Solve the problem and return the plan
        return self.pddl_solver.solve()

    def get_pddl_state(self,gameState:GameState) -> Tuple[List[Tuple],List[Tuple]]:
        """
        This function collects pddl :objects and :init states from simulator gameState.
        """
        # Collect objects and states from the gameState

        states = []
        objects = []


        # Collect available foods on the map
        foodLeft = self.getFood(gameState).asList()
        if len(foodLeft) > 0:
            states.append(("food_available",))
        myPos = gameState.getAgentPosition(self.index)
        myObj = "a{}".format(self.index)
        cloestFoodDist = self.closestFood(myPos,self.getFood(gameState), gameState.getWalls())
        if cloestFoodDist != None and cloestFoodDist <=CLOSE_DISTANCE:
            states.append(("near_food",myObj))

        # Collect capsule states
        capsules = self.getCapsules(gameState)
        if len(capsules) > 0 :
            states.append(("capsule_available",))

        mycapsules = self.getCapsulesYouAreDefending(gameState)
        if len(mycapsules) > 0 :
            states.append(("capsule_available_enemy",))

        for cap in capsules:
            if self.getMazeDistance(cap,myPos) <=CLOSE_DISTANCE:
                states.append(("near_capsule",myObj))
                break
        
        # if less than 100 steps left, influence behaviour
        currTimer = gameState.data.timeleft
        if currTimer < 100:
            states.append(('time_running_out',))

        # Collect winning states
        currentScore = gameState.data.score
        if gameState.isOnRedTeam(self.index):
            if currentScore > 0:
                states.append(("winning",))
            if currentScore> 3:
                states.append(("winning_gt3",))
            if currentScore> 5:
                states.append(("winning_gt5",))
            if currentScore> 8:
                states.append(("winning_gt8",))
            if currentScore> 10:
                states.append(("winning_gt10",))
            if currentScore> 20:
                states.append(("winning_gt20",))
        else:
            if currentScore < 0:
                states.append(("winning",))
            if currentScore < -3:
                states.append(("winning_gt3",))
            if currentScore < -5:
                states.append(("winning_gt5",))
            if currentScore < -8:
                states.append(("winning_gt8",))
            if currentScore < -10:
                states.append(("winning_gt10",))
            if currentScore < -20:
                states.append(("winning_gt20",))


        # Collect team agents states
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents :
            agent_object = "a{}".format(agent_index)
            agent_type = "current_agent" if agent_index == self.index else "ally"
            objects += [(agent_object, agent_type)]

            if agent_index != self.index and self.getMazeDistance(gameState.getAgentPosition(self.index), gameState.getAgentPosition(agent_index)) <= CLOSE_DISTANCE:
                states.append(("near_ally",))

            # collect cooperative predicates to leverage cooperative behaviour
            if agent_index != self.index:
                current_action = MixedAgent.CURRENT_ACTION[agent_index]

                # store the current aaction of the friendly teams agents
                if current_action != {}:

                    states.append((f'{current_action}', agent_object))
                    

            if gameState.getAgentPosition(agent_index)[0] >= self.getEnemyBorder(gameState)[0][0]:
                states.append(('in_enemy_territory', agent_object))
                    
            
            if agent_state.scaredTimer>0:
                states.append(("is_scared",agent_object))

            if agent_state.numCarrying>0:
                states.append(("food_in_backpack",agent_object))
                if agent_state.numCarrying >=20 :
                    states.append(("20_food_in_backpack",agent_object))
                if agent_state.numCarrying >= 15:
                    states.append(("15_food_in_backpack",agent_object))
                if agent_state.numCarrying >=10 :
                    states.append(("10_food_in_backpack",agent_object))
                if agent_state.numCarrying >=5 :
                    states.append(("5_food_in_backpack",agent_object))
                if agent_state.numCarrying >=3 :
                    states.append(("3_food_in_backpack",agent_object))
                
            if agent_state.isPacman:
                states.append(("is_pacman",agent_object))
            
        # Collect enemy agents states
        enemies : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getOpponents(gameState)]
        noisyDistance = gameState.getAgentDistances()
        typeIndex = 1
        for enemy_index, enemy_state in enemies:
            enemy_position = enemy_state.getPosition()
            enemy_object = "e{}".format(enemy_index)
            objects += [(enemy_object, "enemy{}".format(typeIndex))]

            if enemy_state.scaredTimer>0:
                states.append(("is_scared",enemy_object))

            if enemy_position != None:
                for agent_index, agent_state in agents:
                    if self.getMazeDistance(agent_state.getPosition(), enemy_position) <= CLOSE_DISTANCE:
                        states.append(("enemy_around",enemy_object, "a{}".format(agent_index)))
                  
            else:
                if noisyDistance[enemy_index] >=LONG_DISTANCE :
                    states.append(("enemy_long_distance",enemy_object, "a{}".format(self.index)))
                elif noisyDistance[enemy_index] >=MEDIUM_DISTANCE :
                    states.append(("enemy_medium_distance",enemy_object, "a{}".format(self.index)))
                else:
                    states.append(("enemy_short_distance",enemy_object, "a{}".format(self.index)))                                                                                                                                                                                                 


            if enemy_state.isPacman:
                states.append(("is_pacman",enemy_object))
            typeIndex += 1

        # count number of enemy pacman 
        enemy_pacman = sum(p.isPacman == 1 for _,p in enemies)

        # flag disappeared food if food disappeared on there is a food lock and there are enemy pacmen left
        disappeared = self.getFoodDisappeared(gameState)
        if len(disappeared) > 0 or self.disappeared_food_lock != None and enemy_pacman > 0:
            states.append(("food_disappeared",))

        # determine end game actions
        p_diff = self.calculateFoodDifference(gameState)
        
        score_check = currentScore > 0 if gameState.isOnRedTeam(self.index) else currentScore < 0

        # if difference between both sides is  > 60% and have a positive score or winning > 8, go into patrol mode
        # or theres no more food left
        if (p_diff >= 65 and score_check) in states or len(foodLeft) == 0:
            states.append(("patrol_threshold",))

        # if no time left and have 50% more food than enemy side, go dumb aggro
        if currTimer < 80 and p_diff < 50 and not score_check:
            states.append(("get_in_there_buddy",))

        return objects, states
    
    def stateSatisfyCurrentPlan(self, init_state: List[Tuple],positiveGoal, negtiveGoal):
        if self.highLevelPlan is None:
            # No plan, need a new plan
            self.currentNegativeGoalStates = negtiveGoal
            self.currentPositiveGoalStates = positiveGoal
            return False
        
        if positiveGoal != self.currentPositiveGoalStates or negtiveGoal != self.currentNegativeGoalStates:
            return False
        
        if self.pddl_solver.matchEffect(init_state, self.highLevelPlan[self.currentActionIndex][0] ):
            # The current state match the effect of current action, current action action done, move to next action
            if self.currentActionIndex < len(self.highLevelPlan) -1 and self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex+1][0]):
                # Current action finished and next action is applicable
                self.currentActionIndex += 1
                self.lowLevelPlan = [] # reset low level plan
                return True
            else:
                # Current action finished, next action is not applicable or finish last action in the plan
                return False

        if self.pddl_solver.satisfyPrecondition(init_state, self.highLevelPlan[self.currentActionIndex][0]):
            # Current action precondition satisfied, continue executing current action of the plan
            return True
        
        # Current action precondition not satisfied anymore, need new plan
        return False
    
    def getGoals(self, objects: List[Tuple], initState: List[Tuple]):
        # Check a list of goal functions from high priority to low priority if the goal is applicable
        # Return the pddl goal states for selected goal function
        scared_states = []
        in_enemy_territory = []

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                scared_states += [('is_scared', agent_obj)]

            if agent_type == 'ally' or agent_type == 'current_agent':
                in_enemy_territory += [('in_enemy_territory', agent_obj)]

                
        if (("patrol_threshold",) in initState):
            return self.goalDefWinning(objects, initState)

        elif not any(item in in_enemy_territory for item in initState):
            return self.goalRush(objects, initState)

        elif ('food_disappeared',) in initState and ('defence_active',f'a{self.index}') not in initState:
            return self.goalTargetEater(objects, initState)
        
        elif ('15_food_in_backpack', f'a{self.index}') in initState:
            return self.goalConsumeBackpack(objects, initState)
        
        else:
            return self.goalScoring(objects, initState)


    def goalConsumeBackpack(self,objects: List[Tuple], initState: List[Tuple]):
        positiveGoal = []
        negtiveGoal = []
        
        if (("10_food_in_backpack", f'a{self.index}') in initState):
            negtiveGoal += [('is_pacman', f'a{self.index}'), ('10_food_in_backpack', f'a{self.index}')]

        return positiveGoal, negtiveGoal

    def goalRush(self,objects: List[Tuple], initState: List[Tuple]):
        # go to enemy side blindly

        positiveGoal = []
        negtiveGoal = [] # no food avaliable means eat all the food

        #our_agents = []
        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "current_agent":
                positiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.

        
        return positiveGoal, negtiveGoal
    
    def goalScoring(self,objects: List[Tuple], initState: List[Tuple]):
        # If we are not winning more than 5 points,
        # we invate enemy land and eat foods, and bring then back.

        positiveGoal = []
        negtiveGoal = [("food_available",)] # no food avaliable means eat all the food

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.

        return positiveGoal, negtiveGoal
    
    def goalTargetEater(self,objects: List[Tuple], initState: List[Tuple]):

        positiveGoal = []
        negtiveGoal = [("food_disappeared",)] # no food avaliable means eat all the food

        for obj in objects:
            agent_obj = obj[0]
            agent_type = obj[1]
            
            if agent_type == "enemy1" or agent_type == "enemy2":
                negtiveGoal += [("is_pacman", agent_obj)] # no enemy should standing on our land.

        return positiveGoal, negtiveGoal
    
    def goalDefWinning(self,objects: List[Tuple], initState: List[Tuple]):
        # If winning greater than 10 points,
        # this example want defend foods only, and let agents patrol on our ground.
        # The "win_the_game" pddl state is only reachable by the "patrol" action in pddl,
        # using it as goal, pddl will generate plan eliminate invading enemy and patrol on our ground.

        positiveGoal = [("defend_foods",)]
        negtiveGoal = []
        
        return positiveGoal, negtiveGoal

   
    #------------------------------- Heuristic search low level plan Functions -------------------------------
    def getLowLevelPlanHS(self, gameState: GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        # This is a function for plan low level actions using heuristic search.
        # You need to implement this function if you want to solve low level actions using heuristic search.
        # Here, we list some function you might need, read the GameState and CaptureAgent code for more useful functions.
        # These functions also useful for collecting features for Q learnning low levels.

        map = gameState.getWalls() # a 2d array matrix of obstacles, map[x][y] = true means a obstacle(wall) on x,y, map[x][y] = false indicate a free location
        foods = self.getFood(gameState) # a 2d array matrix of food,  foods[x][y] = true if there's a food.
        capsules = self.getCapsules(gameState) # a list of capsules
        foodNeedDefend = self.getFoodYouAreDefending(gameState) # return food will be eatan by enemy (food next to enemy)
        capsuleNeedDefend = self.getCapsulesYouAreDefending(gameState) # return capsule will be eatan by enemy (capsule next to enemy)
        
        #print(self.getBorder(gameState))
        
        # if attacking, strategy is to go for closest food
        # if not on enemy side, first aim to get there. override position if food is close
        # if on enemy side, just keep going for closest food. high level action should handle if enemy is close by or in danger.
        pos = gameState.getAgentPosition(self.index)
        goal = None

        team_index = None
        agents : List[Tuple[int,AgentState]] = [(i,gameState.getAgentState(i)) for i in self.getTeam(gameState)]
        for agent_index, agent_state in agents :
            if self.index != agent_index:
                team_pos = gameState.getAgentPosition(agent_index)
                team_index=agent_index

        # calculate team distance
        team_dist = self.getMazeDistance(pos, team_pos)

        # get friendly position
        path = []
        
        food_density = self.getFoodDensity(foodNeedDefend)

        # early exit actions
        # here are actions we intend to use approximate Q-learning for
        if highLevelAction in ['defence_avoid']:
            self.resetFoodLock()
            path = []
            return path

        # rush to the enemy side.
        if highLevelAction == 'rush':
            self.resetFoodLock()
            border_points = self.getEnemyBorder(gameState)
  
            min_goal_node = self.getGoalPositionByDistanceMetric(pos, border_points, how=np.argmin)

            distance = self.getMazeDistance(pos, min_goal_node)

            # if agent is within 3 tiles of the closest border position, move to there
            if distance <= 3 and team_dist > 3:
                print("Min rush point chosen")
                goal = min_goal_node
            
            else:
                
                goal = random.choice(border_points)
                print(f'Random rush point chosen {goal}')
            
            path = self.run_astar(pos, goal, gameState)
            print(f"pos: {pos}, goal: {goal}")

        elif highLevelAction == 'dive_bomb':
            self.resetFoodLock()
            # dive in head first to the closest food
            food_list = foods.asList()
            goal = self.getGoalPositionByDistanceMetric(pos, food_list, how=np.argmin)

            path = self.run_astar(pos, goal, gameState)

        elif highLevelAction in ['defence_active', 'defence_passive', 'defence_ping'] :
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            enemiesAround = [a for a in enemies if a.isPacman and a.getPosition() != None]
            #invaders = [a for a in enemies if a.isPacman]

            disappeared_foods = self.getFoodDisappeared(gameState)

            if len(enemiesAround) > 0:
                print("get the enemy")
                dists = [self.getMazeDistance(pos, a.getPosition()) for a in enemiesAround]
                #features['enemyDistance'] = min(dists)
                #print(dists)
                goal = min(enemiesAround, key=lambda obj: dists[enemiesAround.index(obj)]).getPosition()
                self.resetFoodLock()
            
            # find closest disappeared food and path find there
            elif len(disappeared_foods) > 0:
                #disappeared_foods
                print("food disappeared")
                goal = self.getGoalPositionByDistanceMetric(pos, disappeared_foods, how=np.argmin)
                self.setFoodLock(goal)

            elif self.disappeared_food_lock != None:
                goal = self.disappeared_food_lock


            # find the closest bottleneck position that isn't already guarded
            # if none, then go to closest food density
            else:
                self.resetFoodLock()
                #print("go to food density")
                bottlenecks = [x for x in self.bottlenecks if x != team_index]
                if len(bottlenecks) >0:
                    goal = self.getGoalPositionByDistanceMetric(pos, bottlenecks, how=np.argmin)

                # if no bottlenecks left that teammate hasnt gone to, go to food density
                else:
                    positions = [key for key, value in food_density.items() if value == max(food_density.values())]
                    goal = self.getGoalPositionByDistanceMetric(pos, positions, how=np.argmin)

            path = self.run_astar(pos, goal, gameState)
            
        # path find to the nearest border point to get away
        elif highLevelAction == 'go_home' or highLevelAction == 'consume_food_quicc':
            self.resetFoodLock()

            # distance calc sourced from https://codereview.stackexchange.com/a/28210
            border_points = self.getBorder(gameState)

            goal = self.getGoalPositionByDistanceMetric(pos, border_points, how=np.argmin)

            path = self.run_astar(pos, goal, gameState, avoid_enemy=True)

        elif highLevelAction == 'patrol':
            self.resetFoodLock()
            print("border patrol")
            # pick the furthest position away on the border location
            border_points = self.getBorder(gameState)

            points_no_team = [x for x in border_points if x != MixedAgent.TARGET_LOCATION[team_index]]
            goal = self.getGoalPositionByDistanceMetric(pos, points_no_team, how=np.argmax)

            print(f'pos: {pos}, goal: {goal}')
            path = self.run_astar(pos, goal, gameState, avoid_enemy_territory=False)

        # else we just choose a random action
        else:
            
            path = []

        if len(path) >0:
            print(path)
            MixedAgent.TARGET_LOCATION[self.index] = goal
        return path # You should return a list of tuple of move action and target location (exclude current location).
    
    def posSatisfyLowLevelPlan(self,gameState: GameState):
        if self.lowLevelPlan == None or len(self.lowLevelPlan)==0 or self.lowLevelActionIndex >= len(self.lowLevelPlan):
            return False
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,self.lowLevelPlan[self.lowLevelActionIndex][0])
        if nextPos != self.lowLevelPlan[self.lowLevelActionIndex][1]:
            return False
        return True

    #------------------------------- Q-learning low level plan Functions -------------------------------

    """
    Iterate through all q-values that we get from all
    possible actions, and return the action associated
    with the highest q-value.
    """
    def getLowLevelPlanQL(self, gameState:GameState, highLevelAction: str) -> List[Tuple[str,Tuple]]:
        values = []
        legalActions = gameState.getLegalActions(self.index)
        rewardFunction = None
        featureFunction = None
        weights = None
        learningRate = 0

        ##########
        # The following classification of high level actions is only a example.
        # You should think and use your own way to design low level planner.
        ##########
        if highLevelAction == "attack":
            # The q learning process for offensive actions are complete, 
            # you can improve getOffensiveFeatures to collect more useful feature to pass more information to Q learning model
            # you can improve the getOffensiveReward function to give reward for new features and improve the trainning process .
            rewardFunction = self.getOffensiveReward
            featureFunction = self.getOffensiveFeatures
            weights = self.getOffensiveWeights()
            learningRate = self.alpha
            
        elif highLevelAction == "go_home":
            # The q learning process for escape actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = self.alpha # learning rate set to 0 as reward function not implemented for this action, do not do q update, 
        
        elif highLevelAction == "consume_food_quicc":

            rewardFunction = self.getEscapeReward
            featureFunction = self.getEscapeFeatures
            weights = self.getEscapeWeights()
            learningRate = 0 # learning rate set to 0 as reward function not implemented for this action, do not do q update, 
        
        # in the end just needed to add a feature to base defence
        elif highLevelAction == 'defence_avoid':
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = self.alpha  

        else:
            # The q learning process for defensive actions are NOT complete,
            # Introduce more features and complete the q learning process
            rewardFunction = self.getDefensiveReward
            featureFunction = self.getDefensiveFeatures
            weights = self.getDefensiveWeights()
            learningRate = self.alpha  

        if len(legalActions) != 0:
            prob = util.flipCoin(self.epsilon) # get change of perform random movement
            if prob and self.trainning:
                action = random.choice(legalActions)
            else:
                for action in legalActions:
                        if self.trainning:
                            self.updateWeights(gameState, action, rewardFunction, featureFunction, weights,learningRate)
                        values.append((self.getQValue(featureFunction(gameState, action), weights), action))
                action = max(values)[1]
        myPos = gameState.getAgentPosition(self.index)
        nextPos = Actions.getSuccessor(myPos,action)
        return [(action, nextPos)]


    """
    Iterate through all features (closest food, bias, ghost dist),
    multiply each of the features' value to the feature's weight,
    and return the sum of all these values to get the q-value.
    """
    def getQValue(self, features, weights):
        return features * weights
    
    """
    Iterate through all features and for each feature, update
    its weight values using the following formula:
    w(i) = w(i) + alpha((reward + discount*value(nextState)) - Q(s,a)) * f(i)(s,a)
    """
    def updateWeights(self, gameState, action, rewardFunction, featureFunction, weights, learningRate):
        features = featureFunction(gameState, action)
        nextState = self.getSuccessor(gameState, action)

        reward = rewardFunction(gameState, nextState)
        for feature in features:
            correction = (reward + self.discountRate*self.getValue(nextState, featureFunction, weights)) - self.getQValue(features, weights)
            weights[feature] =weights[feature] + learningRate*correction * features[feature]
        
    
    """
    Iterate through all q-values that we get from all
    possible actions, and return the highest q-value
    """
    def getValue(self, nextState: GameState, featureFunction, weights):
        qVals = []
        legalActions = nextState.getLegalActions(self.index)

        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                features = featureFunction(nextState, action)
                qVals.append(self.getQValue(features,weights))
            return max(qVals)
    
    def getOffensiveReward(self, gameState: GameState, nextState: GameState):
        # Calculate the reward. 
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        capsules = self.getCapsules(gameState)
        ghosts = self.getGhostLocs(gameState)

        capsules_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(cap,gameState.getWalls()) for cap in capsules)

        ghost_1_step = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(g,gameState.getWalls()) for g in ghosts)

        base_reward =  -50 + nextAgentState.numReturned + nextAgentState.numCarrying
        new_food_returned = nextAgentState.numReturned - currentAgentState.numReturned
        score = self.getScore(nextState)
        
        # get states of the team
        team = [nextState.getAgentState(i) for i in self.getTeam(nextState)]
        num_attackers = [x.isPacman for x in team]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        
        # reward for both attackers being distance 2 away
        if sum(num_attackers) == 2:
            if team_dist >= 2:
                base_reward += 1

        if ghost_1_step > 0 and capsules_1_step == 0:
            base_reward -= 5

        # reward being chased and having a capsule
        if ghost_1_step > 0 and capsules_1_step > 0:
            base_reward += 2
        if score <0:
            base_reward += score
        if new_food_returned > 0:
            # return home with food get reward score
            base_reward += new_food_returned*10
        
        
        print("Agent ", self.index," reward ",base_reward)
        return base_reward
    
    def getDefensiveReward(self,gameState, nextState):
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        # want to remove all invaders, and also bias towards being close to as many of our foods as possible
        # we do not want to be eaten when scared

        score = self.getScore(nextState)

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        
        invaders_1_step_away = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(i, gameState.getWalls()) for i in invaders)
        base_reward = -50 - (len(self.getFoodYouAreDefending(gameState).asList()) - len(self.getFoodYouAreDefending(nextState).asList()))*10
        
        # reward agents for being approximately MEDIUM_DISTANCE away from each other
        # we determine this by sampling as a gaussian centered on MEDIUM_DISTANCE with std = sqrt(LONG_DISTANCE)
        team = [nextState.getAgentState(i) for i in self.getTeam(nextState)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())

        # reward being distant
        if team_dist >= MEDIUM_DISTANCE:
            base_reward += 15

        if invaders_1_step_away >0:
            base_reward += 10

        if score <0:
            print('score')
            base_reward += score

        # compute distance to centre of map
        borders = self.getBorder(gameState)
        centre = borders[len(borders)//2]
        centre_dist = self.getMazeDistance(nextAgentState.getPosition(), centre)
       
        if centre_dist <= CLOSE_DISTANCE:
            base_reward += 5
        
        return base_reward
    
    def getDefAvoidReward(self,gameState, nextState):
        currentAgentState:AgentState = gameState.getAgentState(self.index)
        nextAgentState:AgentState = nextState.getAgentState(self.index)

        # want to remove all invaders, and also bias towards being close to as many of our foods as possible
        # we do not want to be eaten when scared

        #myState = nextAgentState.getAgentState(self.index)
        myPos = nextAgentState.getPosition()

        score = self.getScore(nextState)

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        
        invaders_1_step_away = sum(nextAgentState.getPosition() in Actions.getLegalNeighbors(i, gameState.getWalls()) for i in invaders)
        base_reward = -50 - (len(self.getFoodYouAreDefending(gameState).asList()) - len(self.getFoodYouAreDefending(nextState).asList()))*10
        
        # reward agents for being approximately MEDIUM_DISTANCE away from each other
        # we determine this by sampling as a gaussian centered on MEDIUM_DISTANCE with std = sqrt(LONG_DISTANCE)
        team = [nextState.getAgentState(i) for i in self.getTeam(nextState)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())

        # reward being distant
        if team_dist >= MEDIUM_DISTANCE:
            base_reward += 15

        if invaders_1_step_away >0 and nextAgentState.scaredTimer == 0:
            base_reward += 10

        if score <0:
            print('score')
            base_reward += score

        # compute distance to centre of map
        borders = self.getBorder(gameState)
        centre = borders[len(borders)//2]
        centre_dist = self.getMazeDistance(nextAgentState.getPosition(), centre)
       
        if centre_dist <= CLOSE_DISTANCE:
            base_reward += 5

        # if scared, dont be near invader
        if nextAgentState.scaredTimer <0:
            if len(invaders_1_step_away) > 0:
                base_reward -= 5
        
        return base_reward

    
    def getEscapeReward(self,gameState, nextState):
        print("Warnning: EscapeReward not implemented yet, and learnning rate is 0 for escape",file=sys.stderr)
        return 0



    #------------------------------- Feature Related Action Functions -------------------------------


    
    def getOffensiveFeatures(self, gameState: GameState, action):
        food = self.getFood(gameState) 
        capsules = self.getCapsules(gameState)
        currAgentState = gameState.getAgentState(self.index)

        walls = gameState.getWalls()
        ghosts = self.getGhostLocs(gameState)
        
        # Initialize features
        features = util.Counter()
        nextState = self.getSuccessor(gameState, action)

        # Successor Score
        features['successorScore'] = self.getScore(nextState)/(walls.width+walls.height) * 10

        # Bias
        features["bias"] = 1.0
        
        # Get the location of pacman after he takes the action
        next_x, next_y = nextState.getAgentPosition(self.index)

        # Number of Ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts) 
        
        # Number of ghosts 2-steps away
        #features['#-of-ghosts-2-step-away'] = sum(self.getMazeDistance((next_x, next_y) , g) <= 2 for g in ghosts)

        team = [nextState.getAgentState(i) for i in self.getTeam(nextState)]
        num_attackers = [x.isPacman for x in team]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())

        if sum(num_attackers) == 2:
            features['teamDistance'] = team_dist/(walls.width+walls.height)
        else:
            features['teamDistance'] = 0

        dist_home =  self.getMazeDistance((next_x, next_y), gameState.getInitialAgentPosition(self.index))+1

        features["chance-return-food"] = (currAgentState.numCarrying)*(1 - dist_home/(walls.width+walls.height)) # The closer to home, the larger food carried, more chance return food
        
        # Closest food
        dist = self.closestFood((next_x, next_y), food, walls)
        if dist is not None:
                # make the distance a number less than one otherwise the update
                # will diverge wildly
                features["closest-food"] = dist/(walls.width+walls.height)
        else:
            features["closest-food"] = 0

        # Closest capsule. can use the closestFood function as it performs the same computation
        #print(capsules)
        dist = None
        closest_dist = float('inf')
        #closest_capsule = []
        for cap in capsules:
            dist = self.getMazeDistance(cap, (next_x, next_y))
            if dist < closest_dist:
                closest_dist = dist
                #closest_capsule = cap
            
        if dist is not None:
            features["closest-capsule"] = closest_dist/(walls.width+walls.height)
        else:
            features["closest-capsule"] = 0

        return features

    def getOffensiveWeights(self):
        return MixedAgent.QLWeights["offensiveWeights"]
    
    def getEscapeFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        enemiesAround = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(enemiesAround) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemiesAround]
            features['enemyDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        features["distanceToHome"] = self.getMazeDistance(myPos,self.startPosition)

        return features

    def getEscapeWeights(self):
        return MixedAgent.QLWeights["escapeWeights"]
    
    def getDefAvoidWeights(self):
        return MixedAgent.QLWeights["defAvoid"]

    def getDefAvoidFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        walls = gameState.getWalls()

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        # onDefence acts as the bias for defensive moves
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features['teamDistance'] = team_dist/(walls.width+walls.height)

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)/(walls.width+walls.height)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # compute distance to centre of map
        #centre = walls.width//2, walls.height//2
        borders = self.getBorder(gameState)
        centre = borders[len(borders)//2]
        centre_dist = self.getMazeDistance(successor.getAgentPosition(self.index), centre)
        features['centre_dist'] = centre_dist/(walls.width+walls.height)

        # if ghost is scared, gtfricko
        if myState.scaredTimer > 0:
            if len(invaders) > 0:
                dist = min(self.getMazeDistance(myPos, a.getPosition()) for a in invaders)
                if dist < 3:
                    features['scared-ghost'] = 1
        else:
            features['scared-ghost'] = 0

        return features

    def getDefensiveFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        walls = gameState.getWalls()

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        # onDefence acts as the bias for defensive moves
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        team = [successor.getAgentState(i) for i in self.getTeam(successor)]
        team_dist = self.getMazeDistance(team[0].getPosition(), team[1].getPosition())
        features['teamDistance'] = team_dist/(walls.width+walls.height)

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)/(walls.width+walls.height)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # compute distance to centre of map
        #centre = walls.width//2, walls.height//2
        borders = self.getBorder(gameState)
        centre = borders[len(borders)//2]
        centre_dist = self.getMazeDistance(successor.getAgentPosition(self.index), centre)
        features['centre_dist'] = centre_dist/(walls.width+walls.height)

        # if ghost is scared, gtfricko
        if myState.scaredTimer > 0:
            if len(invaders) > 0:
                dist = min(self.getMazeDistance(myPos, a.getPosition()) for a in invaders)
                if dist < 3:
                    features['scared-ghost'] = 1
        else:
            features['scared-ghost'] = 0    

        return features

    def getDefensiveWeights(self):
        return MixedAgent.QLWeights["defensiveWeights"]
    
    def isAgentRepeatingActions(self):
        """
        Check if an agent is repeating actions. 
        We define this if the agent repeats the same set of actions two times in a row
        eg. if the agent moves from (1,1) to (1,2), then from (1,2) to (1,1), then from (1,1) to (1,2) again
        we consider the agent to repeat actions and flag that this is happening.
        """

        if len(self.observationHistory) > 6:
            
            obs = []
            for hist in self.observationHistory[-6:-1]:
                #print(hist.getAgentPosition(self.index))
                obs.append(hist.getAgentPosition(self.index))

            positions = set()

            for i in range(len(obs) - 1):
                action = hash((obs[i], obs[i+1]))
                if action in positions:
                    return True
                positions.add(action)
        return False

    def calculateFoodDifference(self, gameState):
        """
        Calculate the % difference in food between agent and opponents team.
        """
        oppFoodLeft = len(self.getFood(gameState).asList())
        foodLeft = len(self.getFoodYouAreDefending(gameState).asList())
        if foodLeft > 0:
            percentage_difference = ((foodLeft - oppFoodLeft) / foodLeft) * 100
            return percentage_difference
        else:
            return 0

    def resetFoodLock(self):
        self.disappeared_food_lock = None

    def setFoodLock(self, pos):

        self.disappeared_food_lock = pos

    def countFood(self, food, x_range, y_range):

        """
        Return a count of the number of food dots within the range of x and y values.
        Code adapted from:
        https://github.com/infinityglow/COMP90054-Pacman-Contest-Project/blob/master/myTeam.py#L251
        """

        num_foods = sum(1 for x in x_range for y in y_range if food[x][y])
        return num_foods

    def getFoodDensity(self, food, radius=4):
        """
        Get the density of food for all (x,y) locations in the food data structure.
        The distribution of food over a game map can be simply calculated as 
        the counts of food near a given area.

        Code adapted from:
        https://github.com/infinityglow/COMP90054-Pacman-Contest-Project/blob/master/myTeam.py#L307
        """
        # maintain dictionary of densities
        d = {}
        
        # convert Grid() into list to iterate through easily
        food_list = food.asList()

        # get dimensions of the grid indirectly from food
        width, height = food.width, food.height

        # loop through the food list
        for x, y in food_list:
            x_range = range(max(1, x - radius), min(x + radius, width-1))
            y_range = range(max(1, y - radius), min(y + radius, height-1))
            d[(x,y)] = self.countFood(food, x_range, y_range)
        return d

    def getGoalPositionByDistanceMetric(self, pos, goal_positions, how=np.argmax):
        """
        Calculate the closest or further position from current position pos to all potential goal positions.
        Support numpy arg min or arg max behaviour.
        """
        nodes = np.asarray(goal_positions)
        dist_2 = np.sum((nodes - pos)**2, axis=1)

        goal_idx = how(dist_2)
        goal = tuple(nodes[goal_idx])
        return goal

    def closestFood(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def closestFood_loc(self, pos, food, walls):
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return (pos_x, pos_y)
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def stateClosestFood(self, gameState:GameState):
        pos = gameState.getAgentPosition(self.index)
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        fringe = [(pos[0], pos[1], 0)]
        expanded = set()
        while fringe:
            pos_x, pos_y, dist = fringe.pop(0)
            if (pos_x, pos_y) in expanded:
                continue
            expanded.add((pos_x, pos_y))
            # if we find a food at this location then exit
            if food[pos_x][pos_y]:
                return dist
            # otherwise spread out from the location to its neighbours
            nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
            for nbr_x, nbr_y in nbrs:
                fringe.append((nbr_x, nbr_y, dist+1))
        # no food found
        return None

    def getFoodDisappeared(self, gameState):
        """
        Get location of disappeared food to locate towards
        We find all foods where it was in the previous game state observation but not the current.
        """
        prev_state: GameState = self.getPreviousObservation()
        
        # guarding against not enough history being available
        foods_eaten = []
        if prev_state is not None:
            prev_foods = self.getFoodYouAreDefending(prev_state).asList()
            foods = self.getFoodYouAreDefending(gameState).asList()
            
            for element in prev_foods:
                if element not in foods:
                    foods_eaten.append(element)
            
        return foods_eaten
    
    def getEnemyFoodDisappeared(self, gameState):
        """
        Get location of disappeared food to locate towards
        We find all foods where it was in the previous game state observation but not the current.
        """
        prev_state: GameState = self.getPreviousObservation()
        
        # guarding against not enough history being available
        foods_eaten = []
        if prev_state is not None:
            prev_foods = self.getFood(prev_state).asList()
            foods = self.getFood(gameState).asList()
            
            for element in prev_foods:
                if element not in foods:
                    foods_eaten.append(element)
            
        return foods_eaten  

    def getSuccessor(self, gameState: GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    def getGhostLocs(self, gameState:GameState):
        ghosts = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
                for opponent in opAgents:
                        opPos = gameState.getAgentPosition(opponent)
                        opIsPacman = gameState.getAgentState(opponent).isPacman
                        if opPos and not opIsPacman: 
                                ghosts.append(opPos)
        return ghosts
    
    def getPacmenLocs(self, gameState:GameState):
        pacmen = []
        opAgents = CaptureAgent.getOpponents(self, gameState)
        # Get ghost locations and states if observable
        if opAgents:
            for opponent in opAgents:
                opPos = gameState.getAgentPosition(opponent)
                opIsPacman = gameState.getAgentState(opponent).isPacman
                if opPos and opIsPacman: 
                    pacmen.append(opPos)
        return pacmen

    def getBorder(self, gameState:GameState):
        """
        Get the border line for your  team. 
        if your team is red, this will be half the game board -1. 
        if your team is blue, this will be half
        
        Code adapted from:
        https://github.com/infinityglow/COMP90054-Pacman-Contest-Project/blob/master/myTeam.py


        Parameters
        ----------
        gameState : GameState
            the game state
        """
        # get x coordinate of the border line
        walls = gameState.getWalls()
        x = walls.width // 2

        # red home boarder is one to the left
        if gameState.isOnRedTeam(self.index):
            x -= 1

        border_spaces = [(x, y) for y in range(1, walls.height- 1) if not gameState.hasWall(x, y)]
        return border_spaces
    
    def getEnemyBorder(self, gameState:GameState):
        """
        Get the border line for the enemy team. 
        if your team is red, this will be half the game board. 
        if your team is blue, this will be half-1
        
        Code adapted from:
        https://github.com/infinityglow/COMP90054-Pacman-Contest-Project/blob/master/myTeam.py


        Parameters
        ----------
        gameState : GameState
            the game state
        """
        # get x coordinate of the border line
        walls = gameState.getWalls()
        x = walls.width // 2

        # red home boarder is one to the left. blue to the right
        if not gameState.isOnRedTeam(self.index):
            x -= 1

        border_spaces = [(x, y) for y in range(1, walls.height- 1) if not gameState.hasWall(x, y)]
        return border_spaces
    
    def run_astar(self, start_state, goal_state, gameState: GameState, time_limit_ = sys.maxsize, avoid_enemy=False, avoid_enemy_territory=False):
        """
        Run A* using all instance methods available to the agent.

        will calculate shortest distance while accounting for enemy agent positions.
        just treats enemy agent as static obstacle, and will replan whenever collision is detected.
        since possible paths and states is quite small, can reasonably compute this with multiple replans.
        """
        #def get_path(self,start_state, goal_state):
        nodes_expanded_ = 0
        nodes_generated_ = 0
        open_list_ = bin_heap(compare_node_f)
        all_nodes_list_ = {}
        all_nodes_list_.clear()
        start_time = time.process_time()
        curr_game_state: GameState = gameState.deepCopy()
        start_node = self.generate(start_state, None, None, goal_state, game_state=curr_game_state)
        open_list_.push(start_node)
        all_nodes_list_[start_node] = start_node

        # get all home position
        all_positions = gameState.getWalls().asList(False)
        home_positions = [(x, y) for x, y in all_positions if (x, y) in all_positions and (self.home_turf(x))]

        # continue while there are still nods on OPEN
        while (len(open_list_) > 0):

            #print(f'open list: {open_list_}')
            current: search_node = open_list_.pop()
            current.close()
            #print(f'curr: {current}')
            curr_game_state = current.game_state
            nodes_expanded_ +=1

            # If have time_limit, break time out search.
            if time_limit_ < sys.maxsize:
                runtime_ = time.process_time() - start_time
                if runtime_ > self.time_limit_:
                    status_ = "Time out"
                    return []
                
            # goal example. if successful, return the solution
            if current.state_ == goal_state:
                solution_ = self.solution(current)
                status_ = "Success"
                #self.runtime_ = time.process_time() - self.start_time

                solution_actions = [(x.action_.move_, x.state_) for x in solution_ if x.action_ is not None]
                #return solution_
                return solution_actions

            # expand the current node
            #print(f'all actions: {curr_game_state.getLegalActions(self.index)}')
            for action in curr_game_state.getLegalActions(self.index):
                #print(action)
                succ_action = grid_action(action, 1)
                next_game_state = curr_game_state.generateSuccessor(self.index, action)
                next_pos = next_game_state.getAgentState(self.index).getPosition()
                next_pos = int(next_pos[0]), int(next_pos[1])

                succ_node = self.generate(next_pos, succ_action, current, goal_state, game_state=next_game_state)

                # avoid enemies when planning as static objects
                if avoid_enemy:
                    op_locs = self.getGhostLocs(curr_game_state)

                    # if next position contains an opponent, wiggle around them
                    if next_pos in op_locs:
                        continue
                
                # avoid enemy territory by only considering positions in our territory
                if avoid_enemy_territory:
                    if next_pos not in home_positions:
                        continue

                if succ_node not in all_nodes_list_:
                    # we need this open_handle_ to update the node in open list in the future
                    succ_node.priority_queue_handle_ = open_list_.push(succ_node)
                    all_nodes_list_[succ_node] = succ_node
                    nodes_generated_ += 1

                # succ_node only have the same hash and state comparing with the on in the all nodes list
                # It's not the one in the all nodes list,  we need the real node in the all nodes list.
                exist = all_nodes_list_[succ_node]
                if not exist.is_closed():
                    open_list_ = self.relax(exist, succ_node, open_list_)

        # OPEN list is exhausted and we did not find the goal
        # return failure instead of a solution
        runtime_ = time.process_time() - start_time
        status_ = "Failed"
        return []

    def generate(self, state, action, parent: search_node, goal, heuristic_function_=manhattan_heuristic, heuristic_weight_=1, game_state=None):

        retval = search_node()
        retval.state_ = state
        retval.action_ = action
        retval.game_state = game_state
        if (parent == None):
            # initialise the node from scratch
            # NB: we usually do this only for the start node
            retval.g_ = 0
            retval.depth_ = 0
            retval.timestep_ = 0
        else:
            # initialise the node based on its parent
            retval.g_ = parent.g_ + action.cost_
            retval.depth_ = parent.depth_ + 1
            retval.parent_ = parent
            retval.timestep_= parent.timestep_ + 1


        if heuristic_function_ is None:
            retval.h_ = 0
            retval.f_ = retval.g_
        else:
            retval.h_ = heuristic_function_(state, goal)
            retval.f_ = retval.g_ + retval.h_ * heuristic_weight_
        return retval
    
    def solution(self, goal_node: search_node):
        tmp = goal_node
        depth = goal_node.depth_
        cost = goal_node.g_
        sol = []
        while (tmp != None):
            sol.append(tmp)
            tmp = tmp.parent_

        sol.reverse()
        return sol

    def relax(self, exist:search_node, new:search_node, open_list):
        if exist.g_ > new.g_:
            exist.f_ = new.f_
            exist.g_ = new.g_
            exist.depth_ = new.depth_
            exist.instance_ = new.instance_
            exist.action_ = new.action_
            exist.timestep_ = new.timestep_
            exist.h_ = new.h_
            exist.parent_ = new.parent_
            if exist.priority_queue_handle_ is not None:
                # If handle exist, we are using bin_heap. We need to tell bin_heap one element's value
                # is decreased. Bin_heap will update the heap to maintain priority structure.
                open_list.decrease(exist.priority_queue_handle_)
        return open_list    

    def findBottlenecks(self, gameState):
        """
        Find all the bottlenecks for a given gamestate depending if you are
        on the red or the blue team.
        Code sourced:
        https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L368

        """
        # get the friendly border
        border = self.getBorder(gameState)

        # the starting position is the middle of the border as it theoretically has the most coverage
        # reduces computation time as well
        start = [border[len(border)//2]]
        
        # ending position are all the foods and capsules we are defneding
        end = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)

        # define a flow network with the starting positions
        # dont need to provide ending position here as we do it below
        network, source = self.getFlowNetwork(gameState, start=start)

        # initialise a counter and dictionary to hold the bottlenecks and which foods/capsule they cover
        bottleneckCounter = util.Counter()
        bottleneckPosition = dict()

        # loop through all ending positions
        for pos in end:
            # find all the bottlenecks from the source to the ending position
            bottlenecks = network.FindBottlenecks(source, pos)

            # if there is a bottleneck for this position
            # increment the counter and add to the position dictionary
            if len(bottlenecks) == 1:
                bottleneckCounter[bottlenecks[0]] += 1
                if bottlenecks[0] in bottleneckPosition.keys():
                    bottleneckPosition[bottlenecks[0]].append(pos)
                else:
                    bottleneckPosition[bottlenecks[0]] = [pos]

            # set the flows to 0 for the network
            for edge in network.flow:
                network.flow[edge] = 0

        return bottleneckCounter, bottleneckPosition

    def home_turf(self, x, walls):
            return x < walls.width / 2 if self.red else x >= walls.width / 2

    def getFlowNetwork(self, gameState: GameState, start=None, end=None):
        '''
        Returns the flow network for a given game state using the Ford Fulkerson Algo.
        Takes the starting position as the border positions and the ending position
        as whatever and gets the flow network.

        Code sourced:
        https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L395

        '''
        source = (-1, -1)
        sink = (-2, -2)

        # get the walls and the legal positions of the game state
        # legal positions are the inverse of the walls
        walls = gameState.getWalls()
        legalPositions = gameState.getWalls().asList(False)

        actionPos = lambda x, y: [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        possiblePos = [(x, y) for x, y in legalPositions if (x, y) in legalPositions and (self.home_turf(x))]

        # Make source and sink
        from maxflow import FlowNetwork, SortedEdges
        network = FlowNetwork()

        # Add all vertices
        for pos in possiblePos:
            network.AddVertex(pos)
        network.AddVertex(source)
        network.AddVertex(sink)

        # Add normal edges
        edges = SortedEdges()
        for pos in possiblePos:
            newPos = actionPos(pos[0], pos[1])
            for move in newPos:
                if move in possiblePos:
                    edges[(pos, move)] = 1

        # Add edges from source
        for pos in start or []:
            edges[(source, pos)] = float('inf')

        # Add edges from foods/capsules
        for pos in end or []:
            edges[(pos, sink)] = float('inf')


        # Add edges between all nodes
        for edge in edges:
            network.AddEdge(edge[0], edge[1], edges[edge])

        # retrun the network
        ret = (network,)

        # if the starting pos is not none, add the source to the return value
        if start is not None:
            ret = ret + (source,)
        
        # if the ending position is not none, add the sink node to the return value
        if end is not None:
            ret = tuple(ret) + (sink,)

        return ret
    
    def getTopkBottleneck(self, gameState, k):
        """
        Return the top k number of bottlenecks from the fully solved flow network.
        
        Code adapted from:
        https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L450
        """
        bottleneck_dict, bottleneckPos = self.findBottlenecks(gameState)
        
        # sort the bottleneck dictionary and find the top k bottlenecks by number of food they cover
        top_bottlenecks = sorted(bottleneck_dict.keys(), key=lambda k: bottleneck_dict[k], reverse=True)[:k]
        
        return top_bottlenecks
    

    