from tkinter import *

master = Tk()
triangle_size = 0.1
cell_score_min = -0.2
cell_score_max = 0.2
Width = 100
(x, y) = (5, 5)
actions = ["up", "down", "left", "right"]

board = Canvas(master, width=x*Width, height=y*Width)
player = (0, y-1)
score = 1
restart = False
walk_reward = -0.04

walls = [(1, 1), (1, 2), (2, 1), (2, 2)]
specials = [(4, 1, "red", -1), (4, 0, "green", 1)]
cell_scores = {}
