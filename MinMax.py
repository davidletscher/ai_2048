from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)

		self._nodeCount = 0
		self._parentCount = 0
		self._childCount = 0
		self._depthCount = 0
		self._count = 0

	def findMove(self, state):
		self._count += 1
		actions = self.moveOrder(state)
		depth = 1
		while self.timeRemaining():
			self._depthCount += 1
			self._parentCount += 1
			self._nodeCount += 1
			print('Search depth', depth)
			best = -10000
			for a in actions:
				result = state.move(a)
				if not self.timeRemaining(): return
				v = self.minPlayer(result, depth-1)
				if v is None: return
				if v > best:
					best = v
					bestMove = a
						
			self.setMove(bestMove)
			print('\tBest value', best, bestMove)

			depth += 1

	def maxPlayer(self, state, depth):
		# The max player gets to choose the move
		self._nodeCount += 1
		self._childCount += 1

		if state.gameOver():
			return state.getScore()
			
		actions = self.moveOrder(state)

		if depth == 0:
			return self.heuristic(state)

		self._parentCount += 1
		best = -10000
		for a in actions:
			if not self.timeRemaining(): return None
			result = state.move(a)
			v = self.minPlayer(result, depth-1)
			if v is None: return None
			if v > best:
				best = v
				
		return best

	def minPlayer(self, state, depth):
		# The min player chooses where to add the extra tile and whether it is a 2 or a 4
		self._nodeCount += 1
		self._childCount += 1

		if state.gameOver():
			return state.getScore()
			
		actions = self.moveOrder(state)

		if depth == 0:
			return self.heuristic(state)

		self._parentCount += 1
		best = 1e6
		for (t,v) in state.possibleTiles():
			if not self.timeRemaining(): return None
			result = state.addTile(t,v)
			v = self.maxPlayer(result, depth-1)
			if v is None: return None
			if v < best:
				best = v

		return best

	def heuristic(self, state):
		return state.getScore()
		
	def moveOrder(self, state):
		return state.actions()

	def stats(self):
		print(f'Average depth: {self._depthCount/self._count:.2f}')
		print(f'Branching factor: {self._childCount / self._parentCount:.2f}')
