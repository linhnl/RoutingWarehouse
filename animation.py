import cv2
import numpy as np
from constants import BLOCK_WIDTH, CHANNEL, GRAY
import constants
from routing_map import Routing

def animation(Routing:Routing):

    while True:
        # Routing.Dijkstra_agent()
        Routing.greedy_agent()
        cv2.imshow('frame', draw_map(Routing.Map, (Routing.X, Routing.Y)))
        cv2.waitKey(100)
        
    
def draw_map(map:np.ndarray, current_position:tuple):
    colored_map = np.zeros((map.shape[0]*BLOCK_WIDTH, map.shape[1]*BLOCK_WIDTH, CHANNEL))
    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            match int(map[y, x]):
                case constants.BIN:
                    colored_map[(y*BLOCK_WIDTH): (y*BLOCK_WIDTH)+BLOCK_WIDTH, (x* BLOCK_WIDTH): (x* BLOCK_WIDTH)+BLOCK_WIDTH] = tuple(x/255 for x in constants.BLACK)
                case constants.ROAD:
                    colored_map[(y*BLOCK_WIDTH): (y*BLOCK_WIDTH)+BLOCK_WIDTH, (x* BLOCK_WIDTH): (x* BLOCK_WIDTH)+BLOCK_WIDTH] = tuple(x/255 for x in constants.WHITE)
                case constants.START_POINT:
                    colored_map[(y*BLOCK_WIDTH): (y*BLOCK_WIDTH)+BLOCK_WIDTH, (x* BLOCK_WIDTH): (x* BLOCK_WIDTH)+BLOCK_WIDTH] = tuple(x/255 for x in constants.RED)
                case constants.PICKUP_ITEM:
                    colored_map[(y*BLOCK_WIDTH): (y*BLOCK_WIDTH)+BLOCK_WIDTH, (x* BLOCK_WIDTH): (x* BLOCK_WIDTH)+BLOCK_WIDTH] = tuple(x/255 for x in constants.BLUE)
                case constants.MARKED_ROAD:
                    colored_map[(y*BLOCK_WIDTH): (y*BLOCK_WIDTH)+BLOCK_WIDTH, (x* BLOCK_WIDTH): (x* BLOCK_WIDTH)+BLOCK_WIDTH] = tuple(x/255 for x in constants.ORANGE)
                case _:
                    colored_map[(y*BLOCK_WIDTH): (y*BLOCK_WIDTH)+BLOCK_WIDTH, (x* BLOCK_WIDTH): (x* BLOCK_WIDTH)+BLOCK_WIDTH] = tuple(x/255 for x in constants.GREEN)

    for y in range(map.shape[0]):
        for x in range(map.shape[1]):
            colored_map = cv2.rectangle(colored_map, (x*BLOCK_WIDTH, y*BLOCK_WIDTH), ((x*BLOCK_WIDTH)+BLOCK_WIDTH,(y*BLOCK_WIDTH)+BLOCK_WIDTH), color=GRAY)

    colored_map[(current_position[1]*BLOCK_WIDTH): (current_position[1]*BLOCK_WIDTH)+BLOCK_WIDTH, (current_position[0]* BLOCK_WIDTH): (current_position[0]* BLOCK_WIDTH)+BLOCK_WIDTH] = constants.GREEN

    return colored_map

def block_rectangle(x, y, size):
    return (x, y), (x+size, y), (x+size, y+size), (x, y+size)
