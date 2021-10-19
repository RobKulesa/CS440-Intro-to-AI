# Authors: Robert Kulesa, Daniel Liu, Michael Li
from typing import Optional, List

import numpy as np
import sys
from binheap import BinHeap, State, Cell

sys.setrecursionlimit(10000)


class World:
    def __init__(self, size, grid: np.ndarray = None):
        if grid is None:
            self.grid = np.array([[Cell(j, i) for i in range(size)] for j in range(size)])
        else:
            self.grid = grid

    @staticmethod
    def mhd(row1: int, row2: int, col1: int, col2: int) -> int:
        return np.abs(row2 - row1) + np.abs(col2 - col1)

    @staticmethod
    def mhd_cell(cell1: Cell, cell2: Cell):
        return np.abs(cell2.row - cell1.row) + np.abs(cell2.col - cell1.col)

    def __len__(self):
        return len(self.grid)


class Agent:
    MOVE_NORTH = 0
    MOVE_EAST = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3

    DIRECTIONS = [MOVE_NORTH, MOVE_EAST, MOVE_SOUTH, MOVE_WEST]

    def __init__(self, world: World):
        self.world = world

    # check if move is valid, returning current spot if invalid or new spot if valid. set perform = true to move agent to cell.
    def move(self, start: Cell, direction) -> Optional[Cell]:
        if direction == Agent.MOVE_NORTH:
            if start.row >= 1:
                return self.world.grid[start.row - 1][start.col]
            else:
                return None
        if direction == Agent.MOVE_WEST:
            if start.col >= 1:
                return self.world.grid[start.row][start.col - 1]
            else:
                return None
        if direction == Agent.MOVE_SOUTH:
            if start.row + 1 < len(self.world.grid):
                return self.world.grid[start.row + 1][start.col]
            else:
                return None
        if direction == Agent.MOVE_EAST:
            if start.col + 1 < len(self.world.grid):
                return self.world.grid[start.row][start.col + 1]
            else:
                return None
        return None

    # get unvisited neighbors of the agent's current cell
    def get_unvisited_neighbors(self, cell: Cell, visited: Optional[set]) -> Optional[List[Cell]]:
        res = list()
        for direction in Agent.DIRECTIONS:
            neighbor = self.move(cell, direction)
            if visited is None:
                if neighbor is not None:
                    res.append(neighbor)
            else:
                if neighbor is not None and neighbor not in visited:
                    res.append(neighbor)
        return res

    # initialize gridworld with obstacles using dfs
    def draw_obstacles(self):
        cell = np.random.choice(self.world.grid.flatten())
        cell.type = Cell.EMPTY
        visited = set()
        self.dfs(cell, visited)
        for cell in self.world.grid.flatten():
            if cell not in visited:
                self.dfs(cell, visited)
        for cell in self.world.grid.flatten():
            if cell.type == Cell.NULL:
                cell.type = Cell.OBSTACLE

    def dfs(self, cell: Cell, visited: set):
        visited.add(cell)
        unvisited_neighbors = self.get_unvisited_neighbors(cell, visited)
        while len(unvisited_neighbors) > 0:
            neighbor = np.random.choice(unvisited_neighbors)
            if np.random.randint(0, 10000) <= 3000:
                neighbor.type = Cell.OBSTACLE
                visited.add(neighbor)
            else:
                neighbor.type = Cell.EMPTY
                self.dfs(neighbor, visited)
            unvisited_neighbors.remove(neighbor)

    # Try to compute a path for the agent using forward a star
    def repeated_forward_helper(self, start: Cell, end: Cell, g: list, open_list: BinHeap, visited_blocked: set, search: list, counter: int) -> List[List[Optional[Cell]]]:
        tree: List[List[Optional[Cell]]] = [[None for i in range(len(self.world))] for j in range(len(self.world))]
        while len(open_list) > 0 and g[end.row][end.col] > open_list.peek().get_f_value():
            # Remove a state s with the smallest f-value
            s = open_list.pop_root()
            # Explore neighbors
            neighbors = self.get_unvisited_neighbors(s.cell, visited_blocked)
            for neighbor in neighbors:
                if search[neighbor.row][neighbor.col] < counter:
                    g[neighbor.row][neighbor.col] = np.Infinity
                    search[neighbor.row][neighbor.col] = counter
                if g[neighbor.row][neighbor.col] > g[start.row][start.col] + 1:
                    g[neighbor.row][neighbor.col] = g[start.row][start.col] + 1
                    tree[neighbor.row][neighbor.col] = s.cell
                    idx = open_list.index_of(neighbor)
                    if idx > 0:
                        open_list.remove(idx)
                    open_list.insert(State(neighbor, g[neighbor.row][neighbor.col], World.mhd_cell(neighbor, end)))
        return tree

    def astar(self, start: Cell, end: Cell):
        counter = 0
        search: List[List[int]] = [[0 for i in range(len(self.world))] for j in range(len(self.world))]
        g: List[List[Optional[int]]] = [[None for i in range(len(self.world))] for j in range(len(self.world))]
        visited_blocked = set()
        while not start.__eq__(end):
            counter += 1
            g[start.row][start.col] = 0
            search[start.row][start.col] = counter
            g[end.row][end.col] = np.Infinity
            search[end.row][end.col] = counter
            open_list = BinHeap()
            open_list.insert(State(start, g[start.row][start.col], World.mhd_cell(start, end)))
            tree = self.repeated_forward_helper(start, end, g, open_list, visited_blocked, search, counter)

            if len(open_list) == 0:
                print('I cannot reach the target ;-;')
                return
            # follow tree pointers from goal state to start state, then move agent along resulting path from start state to goal state until it reaches goal state
            #       or one or more action costs on the path increase;
            path: List[Cell] = list()
            path.insert(0, end)
            while path[0].row != start.row or path[0].col != start.col:
                path.insert(0, tree[path[0].row][path[0].col])
            print('path')
            mystr = ''
            for item in path:
                if item is None:
                    mystr = mystr + '(None) '
                elif item.row == end.row and item.col == end.col:
                    mystr = mystr + '[' + str(item.row) + ', ' + str(item.col) + '] '
                else:
                    mystr = mystr + '(' + str(item.row) + ', ' + str(item.col) + ') '
            print(mystr)

            for item in path[1:]:
                if item.type == Cell.OBSTACLE:
                    visited_blocked.add(item)
                    break
                else:
                    start = item

        print('I reached the target! :D')


myWorld = World(6)
myAgent = Agent(myWorld)
myAgent.draw_obstacles()

# obstacles = 0
# emptys = 0
# nulls = 0
# for cell in myWorld.grid.flatten():
#     if cell.type == Cell.EMPTY:
#         emptys += 1
#     if cell.type == Cell.OBSTACLE:
#         obstacles += 1
#     if cell.type == Cell.NULL:
#         nulls += 1
# print('emptys: ', emptys / (emptys + obstacles + nulls), ' obstacles: ', obstacles / (emptys + obstacles + nulls), ' nulls: ', nulls / (emptys + obstacles + nulls))

emptys = [cell for cell in myAgent.world.grid.flatten() if cell.type == Cell.EMPTY]
start, end = np.random.choice(emptys, 2, replace=False)
res = ''
for row in range(len(myAgent.world.grid)):
    for col in range(len(myAgent.world.grid)):
        if start.row == row and start.col == col:
            res += Cell.PLAYER + ' '
        elif end.row == row and end.col == col:
            res += Cell.TARGET + ' '
        else:
            res += str(myAgent.world.grid[row][col]) + ' '
    res += '\n'
print(res)

myAgent.astar(start, end)
