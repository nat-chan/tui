#!/usr/bin/env python
# coding:utf-8
import os
import sys
import numpy as np
#33,65,35
#32,64,35
def first():
	os.system("stty -echo raw")
	print("\033[?1049h"); #画面遷移
	print("\033[?25l");   #カーソル消す
	print("\033[?1002h"); #マウス報告

def last():
	os.system("stty echo -raw")
	print("\033[?1049l");
	print("\033[?25h"); 
	print("\033[?1002l");


#  0
# 1#2
#  3

#  1
# 0#3
#  2

k =' ╴╵┘╷┐│┤╶─└┴┌┬├┼'

row, col = os.get_terminal_size()
table = np.zeros((row, col), dtype=int)
p = np.zeros((2, 2), dtype=int)

first()
try:
	t = b''
	while True:
		t += sys.stdin.buffer.read(1)
		if t[-1] == 3: break
		if t[-6:-3] == b"\033[M":
			if t[-3] == 32:
				p[0] = t[-2]-32, t[-1]-32
				print("\033[{1};{0}H#".format(*p[0]))
			else:
				p[1] = p[0]
				p[0] = t[-2]-32, t[-1]-32
				while np.any(p[0] != p[1]):
					d = p[0] - p[1]
					e = d[0] > d[1], d[0] > -d[1]
					q = 2*e[1] + e[0]

					table[tuple(p[1])] |= 1<<q
					print("\033[{1};{0}H".format(*p[1])+'S')

					p[1, int(e[0]^e[1])] += 2*e[1] - 1
					table[tuple(p[1])] |= 1<<~q&3
					print("\033[{1};{0}H".format(*p[1])+"G")
			t = b''
finally:
	last()

