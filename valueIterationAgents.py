# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    currentIterationCounter = 1
    for state in mdp.getStates():
      self.values[state] = mdp.getReward(state, 'Stop', state)
    while (currentIterationCounter != self.iterations):
      newValues = util.Counter()
      for state in mdp.getStates():
        tempValues = util.Counter()
        for action in mdp.getPossibleActions(state):
          for newStateAndProb in mdp.getTransitionStatesAndProbs(state, action):
            newState = newStateAndProb[0]
            prob = newStateAndProb[1]
            tempValues[action] += prob*(mdp.getReward(state, action, newState)+self.discount*self.values[newState])
        newValues[state] = tempValues[tempValues.argMax()]
      currentIterationCounter += 1
      for state in mdp.getStates():
        self.values[state] = newValues[state]
      
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    returnVal = self.values[state]
    for newStateAndProb in self.mdp.getTransitionStatesAndProbs(state, action):
      newState = newStateAndProb[0]
      prob = newStateAndProb[1]
      returnVal += prob*(self.mdp.getReward(state, action, newState)+self.discount*self.values[newState])
    return returnVal


  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    returnVal = None
    legalActs = self.mdp.getPossibleActions(state)
    if len(legalActs) > 0:
      tempValues = util.Counter()
      for action in legalActs:
        tempValues[action] = self.getQValue(state, action)
      returnVal = tempValues.argMax()
    return returnVal

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
