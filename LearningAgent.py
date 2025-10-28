import pickle
import random
import sys
from math import *
import gzip
import multiprocessing
import time

from Game2048 import *

class Player(BasePlayer):
	def __init__(self, timeLimit):
		BasePlayer.__init__(self, timeLimit)
		
		# Parameters
		self._discountFactor = discountFactor
		
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
			v += self._valueTable[i]
			
		return v

	def findMove(self, board):
		bestValue = float('-inf')
		bestMove = ''
		
		for a in board.actions():
			# Finding the expected (or average) value of the state after the move is taken
			v = 0
			for (result, reward, prob) in board.possibleResults(a):
				v += prob * (reward + self._discountFactor*self.value(result))
				
			if v > bestValue:
				bestValue = v
				bestMove = a
				
		print(f'Best move {bestMove} with value {bestValue}')
		self.setMove(bestMove)
	
# Learning parameters
discountFactor = .999
gamesPerPass = 1000
valueTableSize = 5*16**4
initialLearningRate = .001

def tupleToIndex(t):
	i = 0
	for x in t:
		i = 16*i + x
	return i
	
def tableEntries(board):
	entries = []
	
	for s in board.symmetries():
		entries.append( tupleToIndex(board.getBoard()[0:4]) )
		entries.append( tupleToIndex(board.getBoard()[4:8]) + 16**4 )
		entries.append( tupleToIndex(board.getBoard()[0:2] + board.getBoard()[4:6]) + 2*16**4 )
		entries.append( tupleToIndex(board.getBoard()[1:3] + board.getBoard()[5:7]) + 3*16**4 )
		entries.append( tupleToIndex(board.getBoard()[5:7] + board.getBoard()[9:11]) + 4*16**4 )
	
	return entries

def bestAction(state):
	bestValue = float('-inf')
	bestMove = None
	
	for a in state.actions():
		# Finding the expected (or average) value of the state after the move is taken
		v = 0
		for (result, reward, prob) in state.possibleResults(a):
			v += prob * (reward + sum(valueTable[i] for i in tableEntries(result)))
			
		if v > bestValue:
			bestValue = v
			bestMove = a
			
	return bestMove

def simulateGame(params):
	initialState, learningRate = params
	score = 0
	length = 0
	state = initialState
	episodes = []
	while not state.gameOver():
		a_best = bestAction(state)
		result, reward = state.result(a_best)
		
		score += reward
		length += 1
		maxTile = max(state.getBoard())

		episodes.append((state, result, reward))
			
		state = result
		
	for (state, result, reward) in episodes:
		vState = sum(valueTable[i] for i in tableEntries(state))
		if not result.gameOver():
			vResult = sum(valueTable[i] for i in tableEntries(result))
		else:
			vResult = 0

		error = reward + discountFactor*vResult - vState
		
		for i in tableEntries(state):
			valueTable[i] += learningRate * error
		
	
	return (score, length, maxTile)
		
def train(filename, repetitions):	
	valueTableArray = multiprocessing.RawArray('f', valueTableSize)
	try:
		with gzip.open(filename, 'rb') as dataFile:
			valueTable = pickle.load(dataFile)
		for i in range(valueTableSize):
			valueTableArray[i] = valueTable[i]
		del valueTable
		print('Data loaded')
	except:
		for i in range(valueTableSize):
			valueTableArray[i] = 0.
			
	print('Starting training')
	
	totalGames = 0
	totalLearning = 0
	bestAverageScore = 0
	learningRate = initialLearningRate
	startTime = time.time()
	with multiprocessing.Pool(initializer=initializeThread, initargs=(valueTableArray, )) as pool:
		for rep in range(repetitions):
			# Run gamesPerPass games storing every step in the episode memory
			scores = pool.map(simulateGame, [(Game2048(None,None,True), learningRate) for i in range(gamesPerPass)])
			totalGames += gamesPerPass
			averageScore = sum(s[0] for s in scores)/gamesPerPass
			averageLength = sum(s[1] for s in scores)/gamesPerPass
			totalLearning += sum(s[1] for s in scores)
			maxScore = max(s[0] for s in scores)
			
			maxTileCount = {}
			for s in scores:
				maxTileCount[s[2]] = maxTileCount.get(s[2], 0) + 1				
			maxTiles = ', '.join( f'{2**k} : {maxTileCount[k]}' for k in sorted(list(maxTileCount.keys())) )
			
			print()
			print(f'Average score {averageScore:,.2f}, max score {maxScore:,} and average game length {averageLength:,.2f}.')
			print(f'Max tile count {{ { maxTiles } }}')
			print(f'{totalGames:,} games played and {totalLearning:,} learning steps.')
			print(f'Ellapsed time {time.time() - startTime:.0f} seconds.')

			if averageScore > bestAverageScore or (rep+1) % 50 == 0 or rep == repetitions - 1:
				print('Saving data')
				valueTableCopy = array.array('f', valueTableArray)
				with gzip.open(f'{filename}_{int(totalGames)}_{int(averageScore)}', 'wb') as dataFile:
					pickle.dump(valueTableCopy, dataFile)	
				del valueTableCopy
				bestAverageScore = max(bestAverageScore, averageScore)	
				
def initializeThread(valueTableArray):
	global valueTable
	valueTable = valueTableArray
		
if __name__ == '__main__':
	train(sys.argv[1], int(sys.argv[2]))
