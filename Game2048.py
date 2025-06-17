import time
import random
import copy

class Game2048:
	def __init__(self, b=None, s=None):
		if b:
			self._board = b
		else:
			self._board = [0] * 16
		
		if s:
			self._score = s
		else:
			self._score = 0
		
	def randomize(self):
		self._board = []
		for i in range(16):
			self._board.append(random.choice([0]*16 + [1]*4 + [2]*2 + [3]))

	def actions(self):
		return ''.join([ a for a in 'UDLR' if self.move(a)._board != self._board ])

	def result(self, a):
		s = self._score
		g = self.move(a)
		zeros = [ i for i in range(16) if g._board[i] == 0 ]
		i = random.choice(zeros)
		if random.randint(0,3) == 3:
			g._board[i] = 2
		else:
			g._board[i] = 1
		return g, g._score - s
		
	def getScore(self):
		return self._score
		
	def getTile(self, r, c):
		return self._board[4*r+c]

	def possibleResults(self, a):
		s = self._score
		possible = []
		g = self.move(a)
		zeros = [ i for i in range(16) if g._board[i] == 0 ]
		for i in zeros:
			g = self.move(a)
			for t in [1,2]:
				g._board[i] = t
				if t == 1:
					possible.append((g,.75/len(zeros)))
				else:
					possible.append((g,.25/len(zeros)))
			
		return possible

	def move(self, action):
		board = []
		s = self._score
		if action == 'R':
			for i in range(0,16,4):
				compressed = [t for t in self._board[i:i+4] if t != 0]
				j = len(compressed) - 1
				r = []
				while j >= 0:
					if j > 0 and compressed[j] == compressed[j-1]:
						s += 2*(2**compressed[j])
						r.insert(0,compressed[j]+1)
						j -= 2
					else:
						r.insert(0,compressed[j])
						j -= 1
				r = [0] * (4-len(r)) + r					
				board.extend(r)
			return Game2048(board, s)
		elif action == 'L':
			for i in range(0,16,4):
				compressed = [t for t in self._board[i:i+4] if t != 0]
				j = 0
				r = []
				while j < len(compressed):
					if j < len(compressed)-1 and compressed[j] == compressed[j+1]:
						s += 2*(2**compressed[j])
						r.append(compressed[j]+1)
						j += 2
					else:
						r.append(compressed[j])
						j += 1
				r = r + [0] * (4-len(r))					
				board.extend(r)
			return Game2048(board, s)
		elif action == 'D':
			return self._flip().move('R')._flip()
		elif action == 'U':
			f = self._flip()
			return self._flip().move('L')._flip()
		else:
			print('ERROR move =', action)
				
	def _flip(self):
		r = []
		for i in range(4):
			r.extend( self._board[i:16:4] )
		return Game2048(r, self._score)
		
	def rotate(self, numRotations):
		numRotations = numRotations % 4
		if numRotations == 0:
			return Game2048(copy.copy(self._board), self._score)
			
		if numRotations == 1:
			b = [0]*16
			for r in range(4):
				for c in range(4):
					b[4*c + 3-r] = self._board[4*r+c]
			return Game2048(b, self._score)
			
		if numRotations == 2:
			b = [0]*16
			for r in range(4):
				for c in range(4):
					b[4*(3-r) + 3-c] = self._board[4*r+c]
			return Game2048(b, self._score)
			
		if numRotations == 3:
			b = [0]*16
			for r in range(4):
				for c in range(4):
					b[4*(3-c) + r] = self._board[4*r+c]
			return Game2048(b, self._score)
			
	def gameOver(self):
		return self.actions() == '' or 16 in self._board

	def __str__(self):
		s = ''
		for r in range(0,16,4):
			s += ' '.join(f'{2**x} '.rjust(5) for x in self._board[r:r+4]).replace(' 1 ','   ') + '\n'
		s += f'Score = {self._score}'
		return s
		
class BasePlayer:
	def __init__(self, timeLimit):
		self._timeLimit = timeLimit
		self._startTime = 0
		self._move = None

	def timeRemaining(self):
		if time.time() < self._startTime + self._timeLimit:
			return True
		return False

	def setMove(self, move):
		if self.timeRemaining():
			self._move = move

	def getMove(self):
		return self._move

	def stats(self):
		pass
		
	def saveData(self, filename):
		pass
		
	def loadData(self, filename):
		pass
