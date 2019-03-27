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
    def playGame(self, verbose=False, resume=[]):
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

            self.display(board, valids)
            print("Turn ", str(it), "Player ", str(curPlayer))

            if not(len(resume) > 0 and 'Human' in str(players[curPlayer+1])):
                action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

            if len(resume) > 0:
                action = resume.pop(0)

            if valids[action]==0:
                print(action)
                assert valids[action] >0

            sys.stderr.write(str(action)+',')
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        return self.game.getGameEnded(board, 1)

class MyHumanOthelloPlayer(HumanOthelloPlayer):
    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        if valid[-1]:
            print("Are you sure to pass?")
            sys.stdin.buffer.read(1)
            return self.game.n * self.game.n
#        for i in range(len(valid)):
#            if valid[i]:
#                print(int(i/self.game.n), int(i%self.game.n))
        t = b''
        while True:
            t += sys.stdin.buffer.read(1)
            if t[-1] == 3: break
            if t[-6:-3] == b"\033[M":
                if t[-3] == 32:
                    i = t[-1]-32
                    j = t[-2]-32
                    ii = (i-1)//3
                    jj = (j-1)//5
                    print("\033[{0};{1}H{2},{3}".format(i,j,ii,jj))
                    action = self.game.n * ii + jj
                    if 0 <= action < len(valid) and valid[action]:
                        break
        return action

def mydisplay(board, valids=[0]*65):
    n = board.shape[0]
    tmp = '\033[H\033[30;42m'
    tmp += '┼' + '────┼' * n + '\r\n'
    for i in range(n):
        tmp += '│'
        for j in range(n):
            if board[i][j] == 1:
                tmp += '\033[30m▄██▄│'
            elif board[i][j] == -1:
                tmp += '\033[37m▄██▄\033[30m│'
            elif valids[n*i+j]:
                tmp += '\033[41m    \033[42m│'
            else:
                tmp += '    │'
        tmp += '\r\n│'
        for j in range(n):
            if board[i][j] == 1:
                tmp += '\033[30m▀██▀│'
            elif board[i][j] == -1:
                tmp += '\033[37m▀██▀\033[30m│'
            elif valids[n*i+j]:
                tmp += '\033[41m    \033[42m│'
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
#        rp = RandomPlayer(g).play
#        gp = GreedyOthelloPlayer(g).play
#        hp = HumanOthelloPlayer(g).play
        mhp = MyHumanOthelloPlayer(g).play

# nnet players
        n1 = NNet(g)
        n1.load_checkpoint('../alpha-zero-general-ch/temp1', 'checkpoint_258.pth.tar')
        args1 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
        mcts1 = MCTS(g, n1, args1)
        n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

# nnet players
#        n2 = NNet(g)
#        n2.load_checkpoint('../alpha-zero-general-ch/temp1', 'checkpoint_257.pth.tar')
#        args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
#        mcts2 = MCTS(g, n2, args2)
#        n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

        arena = MyArena(n1p, mhp, g, display=mydisplay)
        result = arena.playGame(verbose=True, resume=[19,34,41,20,37,18,9,43,13,29,51])
    finally:
        last()
        print(result)

if __name__ == '__main__':
    main()
