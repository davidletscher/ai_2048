import random
from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)

	def findMove(self, board):
		actions = board.actions()
		a = random.choice(actions)
		self.setMove(a)
		
