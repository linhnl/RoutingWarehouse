import cv2
import numpy as np
import random
import constants
from constants import BIN, PICKUP_ITEM

class Routing:
    Map: np.ndarray | None
    X: int
    Y: int
    Items: np.ndarray | None
    LastStep: str | None

    def __init__(self, filename) -> None:
        self.load(filename)
        self.LastStep = None
        print(self.Items)

    def get_block_attributes(self, X, Y):
        return self.Map[Y, X]

    def load(self, filename):
        self.Map = np.loadtxt(filename, delimiter='\t')

        start = np.where(self.Map == 1)
        self.X = start[1][0]
        self.Y = start[0][0]

        items = np.where(self.Map == 3)
        list_items = []
        for item in zip(items[0], items[1]):
            list_items.append(item)
        self.Items = np.array(list_items)

    def reload(self):
        items = np.where(self.Map == 3)
        list_items = []
        for item in zip(items[0], items[1]):
            list_items.append(item)
        self.Items = np.array(list_items)
    
    def get_surrounding_blocks(self):
        left   = None if self.X <= 0 else self.Map[self.Y, self.X-1]
        right  = None if self.X >= self.Map.shape[1]-1 else self.Map[self.Y, self.X+1]
        top    = None if self.Y <= 0 else self.Map[self.Y-1, self.X]
        bottom = None if self.Y >= self.Map.shape[0]-1 else self.Map[self.Y+1, self.X]
        return left, right, top, bottom

    def go(self, direction:str):
        match direction:
            case 'left':
                self.X = self.X - 1
            case 'right':
                self.X = self.X + 1
            case 'top':
                self.Y = self.Y - 1
            case 'bottom':
                self.Y = self.Y + 1
    
    def random_agent(self):
        # go random
        left, right, top, bottom = self.get_surrounding_blocks()
        mask = [False if (left in (None, BIN, PICKUP_ITEM)) else True, 
                False if (right in (None, BIN, PICKUP_ITEM)) else True,
                False if (top in (None, BIN, PICKUP_ITEM)) else True,
                False if (bottom in (None, BIN, PICKUP_ITEM)) else True]
        if left == PICKUP_ITEM:
            print(self.Map)
            self.Map[self.Y, self.X-1] == 2
            self.reload()
            print(self.Map)
        if right == PICKUP_ITEM:
            self.Map[self.Y, self.X+1] == 2
            self.reload()
            print(self.Map)
        if top == PICKUP_ITEM:
            self.Map[self.Y-1, self.X] == 2
            self.reload()
            print(self.Map)
        if bottom == PICKUP_ITEM:
            self.Map[self.Y+1, self.X] == 2
            self.reload()
            print(self.Map)

            
        direction = random.choice(np.array(constants.DIRECTION)[mask])
        self.go(direction)
        
    def S_routing_agent(self):
        # go random
        left, right, top, bottom = self.get_surrounding_blocks()
        mask = [False if (left in (None, BIN, PICKUP_ITEM)) else True, 
                False if (right in (None, BIN, PICKUP_ITEM)) else True,
                False if (top in (None, BIN, PICKUP_ITEM)) else True,
                False if (bottom in (None, BIN, PICKUP_ITEM)) else True]
        
        if mask[0] == True:
            direction = 'left'
        
        if mask[0] == True:
            direction = 'left'
        
        if mask[0] == True:
            direction = 'left'
        

        # direction = random.choice(np.array(constants.DIRECTION)[mask])
        self.go(direction)


def load_map(map_path:str):
    map = np.loadtxt(map_path, delimiter='\t')
    return map

def routing(map:np.ndarray, position:tuple):
    current_block = map[position[1]][position[0]]

    return map

def get_surround_block(map:np.ndarray, position:tuple):

    return 