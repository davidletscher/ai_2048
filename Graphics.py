from cs1graphics import *
from Game2048 import *

class Graphics:
	def __init__(self, width):
		scale = width/4

		self._canvas = Canvas(width, width*9//8)
		self._canvas.setTitle('2048')
		self._canvas.setBackgroundColor('tan')
		self._canvas.setAutoRefresh(False)

		self._score = Text('Score: 0')
		self._score.setFontSize(.35*scale)
		self._score.moveTo(.5*width, .25*scale)
		self._canvas.add(self._score)
		
		self._tiles = []
		self._numbers = []
		for i in range(16):
			r = i//4
			c = i%4
			t = Square(scale)
			t.moveTo((c+.5)*scale, (r+1)*scale)
			t.setFillColor('tan')
			t.setBorderColor('tan')
			t.setBorderWidth(.1*scale)
			self._canvas.add(t)
			self._tiles.append(t)
			
			n = Text()
			n.setFontSize(.25*scale)
			n.moveTo((c+.5)*scale, (r+1)*scale)
			self._canvas.add(n)
			self._numbers.append(n)
		
		self._canvas.refresh()
		
		self.colors = []
		for i in range(16):
			self.colors.append(Color())
			
		self.colors[1].setByValue((238, 228, 218))
		self.colors[2].setByValue((237, 224, 200))
		self.colors[3].setByValue((242, 177, 121))
		self.colors[4].setByValue((245, 149, 99))
		self.colors[5].setByValue((246, 124, 95))
		self.colors[6].setByValue((246, 94, 59))
		self.colors[7].setByValue((237, 207, 114))
		self.colors[8].setByValue((237, 204, 97))
		self.colors[9].setByValue((237, 200, 80))
		self.colors[10].setByValue((237, 197, 63))
		self.colors[12].setByValue((237, 194, 46))
		self.colors[13].setByName('black')
		self.colors[14].setByName('black')
		self.colors[15].setByName('black')

	def draw(self, board):		
		for i in range(16):
			if board._board[i] > 0:
				self._tiles[i].setFillColor(self.colors[board._board[i]])
				self._numbers[i].setMessage(str(2**board._board[i]))
			else:
				self._tiles[i].setFillColor('tan')
				self._numbers[i].setMessage('')
		
		
		self._score.setMessage(f'Score: {board._score}')

		self._canvas.refresh()
