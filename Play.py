from Game2048 import *

import sys, importlib, argparse, time

def play(agent, graphicsSize, delay):
	state = Game2048()
	state.randomize()
	if g is not None:
		g.draw(state)

	while not state.gameOver():
		print(state)

		agent._startTime = time.time()
		agent.findMove(state)
		move = agent.getMove()
		
		print()
		print(f'Players moves {move}\n')
		print()
		
		state, reward = state.result(move)

		if g is not None:
			g.draw(state)
			
		if delay:
			time.sleep(delay)
			
	print(state)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description ='Play Othello')
	parser.add_argument('agent', type=str)
	parser.add_argument('time_limit', type=float, help="time to make a move")
	parser.add_argument('-g', type=int, help="size of graphics window")
	parser.add_argument('-t', type=float, help="time delay")
	parser.add_argument('-d', type=str, help="data file")
	args = parser.parse_args()

	try:
		agentModule = importlib.import_module(args.agent.split('.')[0])
	except:
		print('Invalid agent module')
		sys.exit()

	timeLimit = args.time_limit
	agent = agentModule.Player(timeLimit)

	if args.g:
		from Graphics import *
		g = Graphics(args.g)
	else:
		g = None
		
	if args.d:
		agent.loadData(args.d)

	play(agent, g, args.t)
