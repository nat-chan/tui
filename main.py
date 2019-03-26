import sys
sys.path.append('./alpha-zero-general')
import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.chainer.NNet import NNetWrapper as NNet

import numpy as np
from utils import *


def mydisplay(board: np.ndarray):
    n = board.shape[0]
    tmp = '\033[H\033[30;42m'
    tmp += '\n┼' + '────┼' * n + '\n'
    for i in range(n):
        tmp += '│'
        for j in range(n):
            if board[i][j] == 1:
                tmp += '\033[30m▄██▄│'
            elif board[i][j] == -1:
                tmp += '\033[37m▄██▄\033[30m│'
            else:
                tmp += '    │'
        tmp += '\n│'
        for j in range(n):
            if board[i][j] == 1:
                tmp += '\033[30m▀██▀│'
            elif board[i][j] == -1:
                tmp += '\033[37m▀██▀\033[30m│'
            else:
                tmp += '    │'
        tmp += '\n┼' + '────┼' * n + '\n'
    tmp += '\033[0m'
    print(tmp)


g = OthelloGame(8)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('.', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# nnet players
n2 = NNet(g)
n2.load_checkpoint('.', 'best.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, n2p, g, display=mydisplay)
print(arena.playGame(verbose=True))
