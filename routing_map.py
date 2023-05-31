import numpy as np
import random
import constants
from constants import BIN, PICKUP_ITEM, MARKED_ROAD, ROAD
import heapq
import math
from collections import deque

class Routing:
    Map: np.ndarray | None
    X: int
    Y: int
    Items: np.ndarray | None
    LastStep: str | None
    NextItem: np.ndarray | None
    TogoPoints: np.ndarray | None
    PathHelper: np.ndarray | None

    def __init__(self, filename) -> None:
        self.load(filename)
        self.LastStep = None
        # print(self.Items)

    def get_block_attributes(self, X, Y):
        return self.Map[Y, X]

    def load(self, filename):
        self.Map = np.loadtxt(filename, delimiter='\t')

        start = np.where(self.Map == 1)
        self.X = start[1][0]
        self.Y = start[0][0]

        items = np.where(self.Map == 3)
        list_items = []
        list_go_to_point = []
        for item in zip(items[0], items[1]):
            list_items.append(item)
            print("item", item)
            match item[1]:
                case 0, 3, 6:
                    list_go_to_point.append((item[0]+1, item[1]))
                case 2, 5:
                    list_go_to_point.append((item[0]-1, item[1]))

        print("list_go_to_point", list_go_to_point)
        self.Items = np.array(list_items)
        self.TogoPoints = np.array(list_go_to_point)
        self.NextItem, self.PathHelper = self.get_next_item()

    def reload(self):
        self.reverse_marked_road()
        items = np.where(self.Map == 3)
        list_items = []
        for item in zip(items[0], items[1]):
            list_items.append(item)

        if len(list_items) == 0:
            start = np.where(self.Map == 1)
            self.Map[start[0][0], start[1][0]] = 3
            list_items.append((start[0][0], start[1][0]))

        self.Items = np.array(list_items)

        self.NextItem, self.PathHelper = self.get_next_item()
    
    def get_surrounding_blocks(self, block=None):
        if block is not None :
            left   = None if block[1] <= 0 else self.Map[block[0], block[1]-1]
            right  = None if block[1] >= self.Map.shape[1]-1 else self.Map[block[0], block[1]+1]
            top    = None if block[0] <= 0 else self.Map[block[0]-1, block[1]]
            bottom = None if block[0] >= self.Map.shape[0]-1 else self.Map[block[0]+1, block[1]]
            return left, right, top, bottom

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
    
    def pick_item(self):
        left, right, top, bottom = self.get_surrounding_blocks()
        if left == PICKUP_ITEM:
            self.Map[self.Y, self.X-1] = 2
            self.reload()
        if right == PICKUP_ITEM:
            self.Map[self.Y, self.X+1] = 2
            self.reload()
        # if top == PICKUP_ITEM:
        #     self.Map[self.Y-1, self.X] = 2
        #     self.reload()
        # if bottom == PICKUP_ITEM:
        #     self.Map[self.Y+1, self.X] = 2
        #     self.reload()
        # self.Map[self.Y, self.X] = -1

    def reverse_marked_road(self):
        self.Map = np.where(self.Map == MARKED_ROAD, ROAD, self.Map)

    def go_to_start(self):
        start = np.where(self.Map == 1)
        self.NextItem = (start[0][0], start[1][0])
        self.Map[self.NextItem] = 3

    def random_agent(self):
        # go random
        left, right, top, bottom = self.get_surrounding_blocks()
        mask = [False if (left in (None, BIN, PICKUP_ITEM)) else True, 
                False if (right in (None, BIN, PICKUP_ITEM)) else True,
                False if (top in (None, BIN, PICKUP_ITEM)) else True,
                False if (bottom in (None, BIN, PICKUP_ITEM)) else True]
        
        self.pick_item()
            
        direction = random.choice(np.array(constants.DIRECTION)[mask])
        self.go(direction)

    def get_next_item(self):
        # closest_point = []
        # closest_distance = 999
        # for item in self.Items:
        #     distance = euclidean_distance((self.Y, self.X), item)
        #     if distance < closest_distance:
        #         closest_distance = distance
        #         closest_point = item

        nearest_item, nearest_item_path, distance = bfs(self.Map, (self.Y, self.X))
        # print("nearest_item", nearest_item)
        # print("nearest_item_path", nearest_item_path)
        print("distance", distance)

        return nearest_item, np.array(nearest_item_path[:-1])

    def greedy_agent(self):
        self.pick_item()
        if len(self.PathHelper) == 0:
            self.go_to_start()


        next_move = self.PathHelper[0]
        self.PathHelper = self.PathHelper[1:]
        self.Y = next_move[0]
        self.X = next_move[1]
    
    def S_routing_agent(self):
        
        pass
        # direction = random.choice(np.array(constants.DIRECTION)[mask])
        # self.go(direction)

def euclidean_distance(point1, point2):
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance 

def get_valid_neighbors(grid, i, j):
    rows = len(grid)
    cols = len(grid[0])
    neighbors = []
    
    if i > 0 and grid[i - 1][j] != 2 and grid[i - 1][j] != 3:  # Up
        neighbors.append((i - 1, j))
    if i < rows - 1 and grid[i + 1][j] != 2 and grid[i + 1][j] != 3:  # Down
        neighbors.append((i + 1, j))
    if j > 0 and grid[i][j - 1] != 2:  # Left
        neighbors.append((i, j - 1))
    if j < cols - 1 and grid[i][j + 1] != 2:  # Right
        neighbors.append((i, j + 1))
    
    return neighbors


def bfs(grid, start):
    rows = len(grid)
    cols = len(grid[0])
    
    visited = [[False] * cols for _ in range(rows)]
    queue = deque([(start, 0)])
    nearest_item = None
    nearest_item_path = []
    paths = {start: []}
    final_distance = 0
    
    while queue:
        (i, j), distance = queue.popleft()
        visited[i][j] = True
        
        if grid[i][j] == 3:  # Found an item
            nearest_item_path = paths[(i, j)]
            nearest_item = (i, j)
            final_distance = distance
            break
        
        neighbors = get_valid_neighbors(grid, i, j)
        
        for neighbor in neighbors:
            ni, nj = neighbor
            
            if not visited[ni][nj]:
                queue.append(((ni, nj), distance + 1))
                visited[ni][nj] = True
                paths[(ni, nj)] = paths[(i, j)] + [(ni, nj)]
    
    return nearest_item, nearest_item_path, final_distance

