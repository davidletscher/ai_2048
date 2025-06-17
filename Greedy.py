import random
from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)

	def findMove(self, board):
		bestScore = -1000
		bestMove = ''
		
		for a in board.actions():
			print('Testing', a)
			m = board.move(a)
			if m.getScore() > bestScore:
				bestScore = m.getScore()
				bestMove = a
				
		self.setMove(bestMove)
		
