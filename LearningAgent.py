import pickle
import random
import sys
from math import *
import gzip
#import array
import multiprocessing
import time

from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)
		
		# Initialize table
		self._valueTables = []
		self._tableSizes = []
		
		# Parameters
		self._learningRate = .0001
		self._discountFactor = .999
		
		# Setup the table
		self._valueTable = array.array('f', [0.]*valueTableSize)
		
	def loadData(self, filename):
		print('Loading data')
		with gzip.open(filename, 'rb') as dataFile:
			self._valueTables = pickle.load(dataFile)
		
	def saveData(self, filename):
		print('Saving data')
		with gzip.open(filename, 'wb') as dataFile:
			pickle.dump(self._valueTables, dataFile)

	def value(self, board):
		# The table stores the value of the first row.
		# Look at all rotations and add there values so we 
		# also get the last row, first column and last column.
		v = 0.
		for i in tableEntries(board):
			v += self._valueTables[i]
			
		return v / numberOfFeatures

	def findMove(self, board):
		bestValue = float('-inf')
		bestMove = ''
		
		for a in board.actions():
			# Finding the expected (or average) value of the state after the move is taken
			v = 0
			for (result, reward, prob) in board.possibleResults(a):
				v += prob * (reward + self.value(result))
				
			if v > bestValue:
				bestValue = v
				bestMove = a
				
		self.setMove(bestMove)
	
# Learning parameters
discountFactor = .999
episodicMemorySize = 1000000	# Size of episode memory
gamesPerPass = 1000
learningPerPass = 1000000
learningChunkSize = 10000
numberOfFeatures = 24
valueTableSize = 3*16**4

def tupleToIndex(t):
	i = 0
	for x in t:
		i = 16*i + x
	return i
	
def tableEntries(board):
	entries = []
	
	for b in board.symmetries():
		entries.append( tupleToIndex(b.getBoard()[0:4]) )
		entries.append( tupleToIndex(b.getBoard()[4:8]) + 16**4)
		entries.append( tupleToIndex(b.getBoard()[0:2] + b.getBoard()[4:6]) + 2*16**4 )
	
	return entries

class EpisodeMemory:
	def __init__(self, stateArray, rewardArray, resultArray, episodeStart, episodeSize, lock):
		self._state = stateArray
		self._reward = rewardArray
		self._result = resultArray
		
		self._start = episodeStart
		self._size = episodeSize
		
		self._lock = lock
		
	def extend(self, items):
		with self._lock:
			for i, e in enumerate(items):
				if self._size.value < episodicMemorySize:
					end = (self._start.value + self._size.value) % episodicMemorySize
					for i in range(16):
						self._state[16*end+i] = e[0].getBoard()[i]
					self._reward[end] = e[1]
					for i in range(16):
						self._result[16*end+i] = e[2].getBoard()[i]
					self._size.value += 1
				else:
					end = (self._start.value + self._size.value) % episodicMemorySize
					for i in range(16):
						self._state[16*end+i] = e[0].getBoard()[i]
					self._reward[end] = e[1]
					for i in range(16):
						self._result[16*end+i] = e[2].getBoard()[i]
					self._start.value = (self._start.value + 1) % episodicMemorySize
		
	def sample(self, numSamples):
		with self._lock:
			if self._size.value == episodicMemorySize:
				indicies = list(range(episodicMemorySize))
			elif self._start.value + self._size.value <= episodicMemorySize:
				indicies = list(range(self._start.value, self._start.value + self._size.value))
			else:
				indicies = list(range(self._start.value, episodicMemorySize)) + list(range((self._start.value + self._size.value) % episodicMemorySize))
				
			return [ (Game2048(array.array('b',self._state[16*i:16*(i+1)])), self._reward[i], Game2048(array.array('b',self._result[16*i:16*(i+1)]))) for i in random.choices(indicies, k=numSamples) ]


def initializeThread(valueTableArray, stateArray, rewardArray, resultArray, episodeStart, episodeSize, lock):
	global valueTable, episodeMemory
	
	valueTable = valueTableArray
	episodeMemory = EpisodeMemory(stateArray, rewardArray, resultArray, episodeStart, episodeSize, lock)

def simulateGame(initialState):
	score = 0
	length = 0
	transitions = []
	state = initialState
	while not state.gameOver():
		bestValue = float('-inf')
		
		for a in state.actions():
			# Finding the expected (or average) value of the state after the move is taken
			v = 0
			for (result, reward, prob) in state.possibleResults(a):
				v += prob * (reward + discountFactor*sum(valueTable[i] for i in tableEntries(result)))/numberOfFeatures
				
			if v > bestValue:
				bestValue = v
				move = a	
		
		oldState = state
		state, reward = state.result(move)
		transitions.append( (oldState, reward, state) )
			
		length += 1
		score += reward
		
	maxTile = max(state.getBoard())
		
	episodeMemory.extend(transitions)
	
	return (score, length, maxTile)
	
def learning(params):
	repetitions, learningRate = params
	totalError = 0.
	for episode in episodeMemory.sample(repetitions):
		state, reward, result = episode
		
		stateValue = sum(valueTable[i] for i in tableEntries(state))/numberOfFeatures
		resultValue = sum(valueTable[i] for i in tableEntries(result))/numberOfFeatures
		error = reward + discountFactor*resultValue - stateValue
		totalError += abs(error)
		
		for i in tableEntries(state):
			valueTable[i] += learningRate * error / numberOfFeatures
			
	return totalError
		
		
def train(filename, repetitions):
	print('Starting training')
	
	valueTableArray = multiprocessing.RawArray('f', valueTableSize)
	try:
		with gzip.open(filename, 'rb') as dataFile:
			valueTable = pickle.load(dataFile)		
		for i in range(valueTableSize):
			valueTableArray = valueTable[i]
		del valueTable
		print('Data loaded')
	except:
		for i in range(valueTableSize):
			valueTableArray[i] = 0.
			
	stateArray = multiprocessing.RawArray('b', 16*episodicMemorySize)
	rewardArray = multiprocessing.RawArray('i', episodicMemorySize)
	resultArray = multiprocessing.RawArray('b', 16*episodicMemorySize)
	episodeStart = multiprocessing.RawValue('i', 0)
	episodeSize = multiprocessing.RawValue('i', 0)
	lock = multiprocessing.Lock()
	
	totalGames = 0
	totalLearning = 0
	bestAverageScore = 0
	learningRate = .1
	startTime = time.time()
	with multiprocessing.Pool(initializer=initializeThread, initargs=(valueTableArray, stateArray, rewardArray, resultArray, episodeStart, episodeSize, lock)) as pool:
		for rep in range(repetitions):
			# Run 1000 games storing every step in the episode memory
			scores = pool.map(simulateGame, [Game2048(None,None,True) for i in range(gamesPerPass)])
			totalGames += gamesPerPass
			averageScore = sum(s[0] for s in scores)/gamesPerPass
			averageLength = sum(s[1] for s in scores)/gamesPerPass
			maxScore = max(s[0] for s in scores)
			
			maxTileCount = {}
			for s in scores:
				maxTileCount[s[2]] = maxTileCount.get(s[2], 0) + 1				
			maxTiles = ', '.join( f'{2**k} : {maxTileCount[k]}' for k in sorted(list(maxTileCount.keys())) )
			
			print()
			print(f'Average score {averageScore:,.2f}, max score {maxScore:,} and average game length {averageLength:,.2f}.')
			print(f'Max tile count {{ { maxTiles } }}')

			if averageScore > bestAverageScore:
				print('Saving data')
				valueTableCopy = array.array('f', valueTableArray)
				with gzip.open(F'{filename}_{int(totalGames)}_{int(averageScore)}', 'wb') as dataFile:
					pickle.dump(valueTableCopy, dataFile)	
				del valueTableCopy
				bestAverageScore = averageScore				
			
			# Do a million learning passes broken into chunks of size 
			totalError = sum(pool.map(learning, [ (learningChunkSize, learningRate) for i in range(learningPerPass//learningChunkSize)])) / learningPerPass
			totalLearning += learningPerPass
		
			print(f'After {totalGames:,} games and {totalLearning:,.0f} learning passes the average error is {totalError:.2f}')
			print(f'Ellapsed time {time.time() - startTime:.0f} seconds.')
	
	with gzip.open(filename, 'wb') as dataFile:
		valueTableCopy = array.array('f', valueTableArray)
		with gzip.open(F'{filename}_{int(totalLearning/1e6)}_{int(totalError)}', 'wb') as dataFile:
			pickle.dump(valueTableCopy, dataFile)	
		del valueTableCopy
		
if __name__ == '__main__':
	train(sys.argv[1], int(sys.argv[2]))
