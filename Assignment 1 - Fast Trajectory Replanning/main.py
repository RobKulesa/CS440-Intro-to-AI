# Authors: Robert Kulesa, Daniel Liu, Michael Li
from typing import Optional, List

import numpy as np
import sys
from binheap import BinHeap, State, Cell

sys.setrecursionlimit(10000)


class World:
    def __init__(self, size):
        self.grid = np.array([[Cell(j, i) for i in range(size)] for j in range(size)])
        self.target_row = 0
        self.target_col = 0

    def set_target_cell(self, cell: Cell):
        self.target_row = cell.row
        self.target_col = cell.col
        self.grid[cell.row][cell.col].type = Cell.TARGET

    # return the manhattan distance of a cell to the target cell
    def mhd_to_target(self, row: int, col: int) -> int:
        return np.abs(self.target_row - row) + np.abs(self.target_col - col)

    def get_target_cell(self) -> Cell:
        return self.grid[self.target_row][self.target_col]

    def __len__(self):
        return len(self.grid)


class Agent:
    MOVE_NORTH = 0
    MOVE_EAST = 1
    MOVE_SOUTH = 2
    MOVE_WEST = 3

    DIRECTIONS = [MOVE_NORTH, MOVE_EAST, MOVE_SOUTH, MOVE_WEST]

    def __init__(self, world: World, row, col):
        self.world = world
        self.row = row
        self.col = col

    def full_setup(self):
        self.draw_obstacles()
        self.set_agent_and_target_cells()

    def set_agent_cell(self, cell: Cell):
        self.row = cell.row
        self.col = cell.col

    def get_agent_cell(self) -> Cell:
        return self.world.grid[self.row][self.col]

    def get_agent_mhd(self) -> int:
        return self.world.mhd_to_target(self.row, self.col)

    def mhd_to_agent(self, row: int, col: int) -> int:
        return np.abs(self.row - row) + np.abs(self.col - col)

    # check if move is valid, returning current spot if invalid or new spot if valid. set perform = true to move agent to cell.
    def move(self, direction, start: Optional[Cell], perform: bool = True) -> Optional[Cell]:
        if start is None:
            if direction == Agent.MOVE_NORTH:
                if self.row >= 1 and self.world.grid[self.row - 1][self.col].can_move_to():
                    if perform:
                        self.row -= 1
                    return self.world.grid[self.row - 1][self.col]
                else:
                    return None
            if direction == Agent.MOVE_WEST:
                if self.col >= 1 and self.world.grid[self.row][self.col - 1].can_move_to():
                    if perform:
                        self.col -= 1
                    return self.world.grid[self.row][self.col - 1]
                else:
                    return None
            if direction == Agent.MOVE_SOUTH:
                if self.row + 1 < len(self.world.grid) and self.world.grid[self.row + 1][self.col].can_move_to():
                    if perform:
                        self.row += 1
                    return self.world.grid[self.row + 1][self.col]
                else:
                    return None
            if direction == Agent.MOVE_EAST:
                if self.col + 1 < len(self.world.grid) and self.world.grid[self.row][self.col + 1].can_move_to():
                    if perform:
                        self.col += 1
                    return self.world.grid[self.row][self.col + 1]
                else:
                    return None
        else:
            if direction == Agent.MOVE_NORTH:
                if start.row >= 1 and self.world.grid[start.row - 1][start.col].can_move_to():
                    if perform:
                        self.row = start.row - 1
                    return self.world.grid[start.row - 1][start.col]
                else:
                    return None
            if direction == Agent.MOVE_WEST:
                if start.col >= 1 and self.world.grid[start.row][start.col - 1].can_move_to():
                    if perform:
                        self.col = start.col - 1
                    return self.world.grid[start.row][start.col - 1]
                else:
                    return None
            if direction == Agent.MOVE_SOUTH:
                if start.row + 1 < len(self.world.grid) and self.world.grid[start.row + 1][start.col].can_move_to():
                    if perform:
                        self.row = start.row + 1
                    return self.world.grid[start.row + 1][start.col]
                else:
                    return None
            if direction == Agent.MOVE_EAST:
                if start.col + 1 < len(self.world.grid) and self.world.grid[start.row][start.col + 1].can_move_to():
                    if perform:
                        self.col = start.col + 1
                    return self.world.grid[start.row][start.col + 1]
                else:
                    return None
        return None

    def move_to_cell(self, cell: Cell) -> None:
        self.row = cell.row
        self.col = cell.col

    # get unvisited neighbors of the agent's current cell
    def get_unvisited_neighbors(self, cell: Optional[Cell], visited: set) -> Optional[List[Cell]]:
        res = list()
        for direction in Agent.DIRECTIONS:
            neighbor = self.move(direction, cell, False)
            if neighbor is not None and neighbor not in visited:
                res.append(neighbor)
        return res

    # initialize gridworld with obstacles using dfs
    def draw_obstacles(self):
        cell = np.random.choice(self.world.grid.flatten())
        visited = set()
        self.dfs(cell, visited)
        for cell in self.world.grid.flatten():
            if cell not in visited:
                self.dfs(cell, visited)

    def dfs(self, cell: Cell, visited: set):
        self.set_agent_cell(cell)
        visited.add(cell)
        unvisited_neighbors = self.get_unvisited_neighbors(None, visited)
        while len(unvisited_neighbors) > 0:
            neighbor = np.random.choice(unvisited_neighbors)
            if np.random.randint(0, 10000) <= 2200:
                neighbor.type = Cell.OBSTACLE
                visited.add(neighbor)
            else:
                self.dfs(neighbor, visited)
            unvisited_neighbors.remove(neighbor)

    def set_agent_and_target_cells(self):
        emptys = [cell for cell in self.world.grid.flatten() if cell.type == Cell.EMPTY]
        idx = np.random.randint(0, len(emptys) - 1)
        self.set_agent_cell(emptys.pop(idx))
        self.world.set_target_cell(np.random.choice(emptys))

    # return unvisited neighbor with lowest manhattan distance to target cell, with random tie-breaking
    def get_neighbor_lowest_mhd(self, visited: set) -> Optional[Cell]:
        neighbors = self.get_unvisited_neighbors(None, visited)
        if len(neighbors) < 1:
            return None
        res = neighbors[0]
        if len(neighbors) > 1:
            for neighbor in neighbors[1:]:
                if self.world.mhd_to_target(neighbor.row, neighbor.col) < self.world.mhd_to_target(res.row, res.col):
                    res = neighbor
                elif self.world.mhd_to_target(res.row, res.col) == self.world.mhd_to_target(neighbor.row, neighbor.col):
                    res = np.random.choice([res, neighbor])
        return res

    def __str__(self):
        res = str()
        for row in range(len(self.world.grid)):
            for col in range(len(self.world.grid)):
                if row == self.row and col == self.col:
                    res += Cell.PLAYER + ' '
                else:
                    res += str(self.world.grid[row][col]) + ' '
            res += '\n'
        return res

    # Try to compute a path for the agent using forward a star
    def repeated_forward_helper(self, g: list, open_list: BinHeap, visited: set, search: list, counter: int) -> List[List[Optional[Cell]]]:
        tree: List[List[Optional[Cell]]] = [[None for i in range(len(self.world))] for j in range(len(self.world))]
        while len(open_list) > 0 and g[self.world.target_row][self.world.target_col] > open_list.peek().f_value:
            # Remove a state s with the smallest f-value
            s = open_list.pop_root()
            visited.add(s.cell)
            # Explore unvisited neighbors
            neighbors = self.get_unvisited_neighbors(s.cell, visited)
            for neighbor in neighbors:
                if search[neighbor.row][neighbor.col] < counter:
                    g[neighbor.row][neighbor.col] = np.Infinity
                    search[neighbor.row][neighbor.col] = counter
                if g[neighbor.row][neighbor.col] > g[self.row][self.col] + 1:
                    g[neighbor.row][neighbor.col] = g[self.row][self.col] + 1
                    tree[neighbor.row][neighbor.col] = s.cell
                    idx = open_list.index_of(neighbor)
                    if idx > 0:
                        open_list.arr[idx].f_value = g[neighbor.row][neighbor.col] + self.world.mhd_to_target(neighbor.row, neighbor.col)
                        open_list.heapify_up(idx)
                    else:
                        open_list.insert(State(neighbor, g[neighbor.row][neighbor.col] + self.world.mhd_to_target(neighbor.row, neighbor.col)))
                        # print('inserted ', str(neighbor.row), ', ', str(neighbor.col))
        return tree

    def repeated_forward_astar(self):
        counter = 0
        search: List[List[int]] = [[0 for i in range(len(self.world))] for j in range(len(self.world))]
        g: List[List[Optional[int]]] = [[None for i in range(len(self.world))] for j in range(len(self.world))]
        while self.row != self.world.target_row and self.col != self.world.target_col:
            counter += 1
            g[self.row][self.col] = 0
            search[self.row][self.col] = counter
            g[self.world.target_row][self.world.target_col] = np.Infinity
            search[self.world.target_row][self.world.target_col] = counter
            open_list = BinHeap()
            open_list.insert(State(self.get_agent_cell(), g[self.row][self.col] + self.get_agent_mhd()))
            visited = set()
            tree = self.repeated_forward_helper(g, open_list, visited, search, counter)

            # print('tree')
            # for row in tree:
            #     mystr = str()
            #     for item in row:
            #         if item is None:
            #             mystr = mystr + '(None) '
            #         else:
            #             mystr = mystr + '(' + str(item.row) + ', ' + str(item.col) + ') '
            #     print(mystr)

            if len(open_list) == 0:
                print('I cannot reach the target ;-;')
                return
            # follow tree pointers from goal state to start state, then move agent along resulting path from start state to goal state until it reaches goal state
            #       or one or more action costs on the path increase;
            path: List[Cell] = list()
            path.insert(0, self.world.get_target_cell())
            while path[0].row != self.row or path[0].col != self.col:
                path.insert(0, tree[path[0].row][path[0].col])

            print('path')
            mystr = ''
            for item in path:
                if item is None:
                    mystr = mystr + '(None) '
                elif item.row == self.world.target_row and item.col == self.world.target_col:
                    mystr = mystr + '[' + str(item.row) + ', ' + str(item.col) + '] '
                else:
                    mystr = mystr + '(' + str(item.row) + ', ' + str(item.col) + ') '
            print(mystr)

            self.move_to_cell(path[len(path) - 1])

            # update the increased action costs (if any);

        print('I reached the target! :D')


myWorld = World(25)
myAgent = Agent(myWorld, 0, 0)
myAgent.full_setup()
# obstacles = 0
# emptys = 0
# for cell in myWorld.grid.flatten():
#     if cell.type == Cell.EMPTY:
#         emptys += 1
#     if cell.type == Cell.OBSTACLE:
#         obstacles += 1
# print('emptys: ', emptys / (emptys + obstacles), ' obstacles: ', obstacles / (emptys + obstacles))

# neighbor = myAgent.get_neighbor_lowest_mhd(set())
# if neighbor is not None:
#     # neighbor.type = '-'
#     print(neighbor.row, neighbor.col)
# else:
#     print("invalid neighbor")
print(myAgent)

myAgent.repeated_forward_astar()
