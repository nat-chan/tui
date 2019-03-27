import sys
sys.path.append('./alpha-zero-general')
import os
import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.chainer.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

def first():
    os.system("stty -echo raw") #入力文字を非表示
    print("\033[?1049h"); #画面遷移
    print("\033[?25l");   #カーソル消す
    print("\033[?1002h"); #マウス報告

def last():
    os.system("stty echo -raw")
    print("\033[?1049l");
    print("\033[?25h"); 
    print("\033[?1002l");


class MyArena(Arena.Arena):
    def playGame(self, verbose=False):
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1

            print("Turn ", str(it), "Player ", str(curPlayer))
            self.display(board)

            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
        self.display(board)

        return self.game.getGameEnded(board, 1)

def mydisplay(board: np.ndarray):
    n = board.shape[0]
    tmp = '\033[H\033[30;42m'
    tmp += '\r\n┼' + '────┼' * n + '\r\n'
    for i in range(n):
        tmp += '│'
        for j in range(n):
            if board[i][j] == 1:
                tmp += '\033[30m▄██▄│'
            elif board[i][j] == -1:
                tmp += '\033[37m▄██▄\033[30m│'
            else:
                tmp += '    │'
        tmp += '\r\n│'
        for j in range(n):
            if board[i][j] == 1:
                tmp += '\033[30m▀██▀│'
            elif board[i][j] == -1:
                tmp += '\033[37m▀██▀\033[30m│'
            else:
                tmp += '    │'
        tmp += '\r\n┼' + '────┼' * n + '\r\n'
    tmp += '\033[0m'
    print(tmp)

def main():
    first()
    try:
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
        n2.load_checkpoint('../alpha-zero-general-ch/temp', 'checkpoint_104.pth.tar')
        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts2 = MCTS(g, n2, args2)
        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

        arena = MyArena(gp, n1p, g, display=mydisplay)
        result = arena.playGame(verbose=True)
    finally:
        last()
        print(result)

if __name__ == '__main__':
    main()
