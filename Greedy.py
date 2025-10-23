import random
from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)

	def findMove(self, board):
		bestScore = -1000
		bestMove = ''
		
		for a in board.actions():
			avgScore = 0.
			for (result, reward, prob) in board.possibleResults(a):
				avgScore += prob*reward

			if avgScore > bestScore:
				bestScore = avgScore
				bestMove = a
				
		self.setMove(bestMove)
		
