import random
from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)

	def findMove(self, board):
		actions = board.actions()
		if 'R' in actions:
			self.setMove('R')
		elif 'U' in actions:
			self.setMove('U')
		elif 'D' in actions:
			self.setMove('D')
		else:
			self.setMove('L')
		
