import numpy as np
import random
import constants
from constants import BIN, PICKUP_ITEM, MARKED_ROAD, ROAD
import heapq
import math

class Routing:
    Map: np.ndarray | None
    X: int
    Y: int
    Items: np.ndarray | None
    LastStep: str | None
    NextItem: np.ndarray | None

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
        self.NextItem = self.get_next_item()

    def reload(self):
        items = np.where(self.Map == 3)
        list_items = []
        for item in zip(items[0], items[1]):
            list_items.append(item)
        self.Items = np.array(list_items)
        self.reverse_marked_road()
        self.NextItem = self.get_next_item()
    
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
        if top == PICKUP_ITEM:
            self.Map[self.Y-1, self.X] = 2
            self.reload()
        if bottom == PICKUP_ITEM:
            self.Map[self.Y+1, self.X] = 2
            self.reload()
        self.Map[self.Y, self.X] = -1

    def reverse_marked_road(self):
        self.Map = np.where(self.Map == MARKED_ROAD, ROAD, self.Map)

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
        closest_point = []
        closest_distance = 999
        for item in self.Items:
            distance = euclidean_distance((self.Y, self.X), item)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = item

        return closest_point

    def greedy_agent(self):
        # go random
        left, right, top, bottom = self.get_surrounding_blocks()
        mask = [False if (left in (None, BIN, PICKUP_ITEM, MARKED_ROAD)) else True, 
                False if (right in (None, BIN, PICKUP_ITEM, MARKED_ROAD)) else True,
                False if (top in (None, BIN, PICKUP_ITEM, MARKED_ROAD)) else True,
                False if (bottom in (None, BIN, PICKUP_ITEM, MARKED_ROAD)) else True]
        
        self.pick_item()
        

        closest_point = self.NextItem

        optimistic_direction = []
        if self.X < closest_point[1]:
            optimistic_direction.append('right')
        if self.X > closest_point[1]:
            optimistic_direction.append('left')
        if self.Y < closest_point[0]:
            optimistic_direction.append('bottom')
        if self.Y > closest_point[0]:
            optimistic_direction.append('top')

        print(closest_point)

        selected_direction = np.array(list(set(optimistic_direction) & set(np.array(constants.DIRECTION)[mask])))
        selected_direction = np.array(constants.DIRECTION)[mask] if len(selected_direction) == 0 else selected_direction
        direction = random.choice(selected_direction)
        self.go(direction)

    def true_distance(self, direction:np.ndarray):
        
        return 999
    

    def direction_weight(self, directions:np.ndarray):
        for direction in directions:
            if direction == "top":
                self.get_next_item()
        return 999

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

    def Dijkstra_agent(self):
        start = (self.Y, self.X)
        distance, q = bfs(self.Map, start)
        if distance != -1:
            print(f"Distance to the nearest item: {distance}")
            print(f"queue : {q}")
        else:
            print("No items found")

def euclidean_distance(point1, point2):
    distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance 

def dijkstra(grid, start):
    rows = len(grid)
    cols = len(grid[0])
    
    distances = [[float('inf')] * cols for _ in range(rows)]
    distances[start[0]][start[1]] = 0
    
    queue = [(0, start)]
    
    while queue:
        current_distance, (i, j) = heapq.heappop(queue)
        
        if current_distance > distances[i][j]:
            continue
        
        neighbors = get_valid_neighbors(grid, i, j)
        
        for neighbor in neighbors:
            ni, nj = neighbor
            distance = current_distance + 1  # Assuming all path distances are 1
            
            if distance < distances[ni][nj]:
                distances[ni][nj] = distance
                heapq.heappush(queue, (distance, (ni, nj)))
    
    return distances

def get_valid_neighbors(grid, i, j):
    rows = len(grid)
    cols = len(grid[0])
    neighbors = []
    
    if i > 0 and grid[i - 1][j] != 2:  # Up
        neighbors.append((i - 1, j))
    if i < rows - 1 and grid[i + 1][j] != 2:  # Down
        neighbors.append((i + 1, j))
    if j > 0 and grid[i][j - 1] != 2:  # Left
        neighbors.append((i, j - 1))
    if j < cols - 1 and grid[i][j + 1] != 2:  # Right
        neighbors.append((i, j + 1))
    
    return neighbors


def calculate_distance_between_items(grid):
    rows = len(grid)
    cols = len(grid[0])
    item_locations = []
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 3:
                item_locations.append((i, j))
    
    total_distance = 0
    
    for i in range(len(item_locations) - 1):
        start = item_locations[i]
        distances = dijkstra(grid, start)
        end = item_locations[i + 1]
        total_distance += distances[end[0]][end[1]]
    
    return total_distance

from collections import deque

def bfs(grid, start):
    rows = len(grid)
    cols = len(grid[0])
    
    visited = [[False] * cols for _ in range(rows)]
    queue = deque([(start, 0)])
    
    while queue:
        (i, j), distance = queue.popleft()
        visited[i][j] = True
        
        if grid[i][j] == 3:  # Found an item
            return distance, queue
        
        neighbors = get_valid_neighbors(grid, i, j)
        
        for neighbor in neighbors:
            ni, nj = neighbor
            
            if not visited[ni][nj]:
                queue.append(((ni, nj), distance + 1))
                visited[ni][nj] = True
    
    return -1  # No items found