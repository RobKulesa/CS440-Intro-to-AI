# Authors: Robert Kulesa, Daniel Liu, Michael Li
import math
from typing import Optional, List

import numpy as np
import datetime as dt
import sys
from binheap import BinHeap, State, Cell

sys.setrecursionlimit(10000)


class World:
    def __init__(self, size, grid: np.ndarray = None):
        if grid is None:
            self.grid = np.array([[Cell(j, i) for i in range(size)] for j in range(size)])
        else:
            self.grid = np.array([[Cell(j, i) for i in range(len(grid))] for j in range(len(grid))])
            for row in range(len(grid)):
                for col in range(len(grid)):
                    self.grid[row][col].type = grid[row][col]

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

    def print_world(self, start, end):
        res = ''
        for row in range(len(self.world.grid)):
            for col in range(len(self.world.grid)):
                if start.row == row and start.col == col:
                    res += Cell.PLAYER + ' '
                elif end.row == row and end.col == col:
                    res += Cell.TARGET + ' '
                else:
                    res += str(self.world.grid[row][col]) + ' '
            res += '\n'
        print(res)

    def reset(self):
        for item in self.world.grid.flatten():
            if item.type == Cell.PATH:
                item.type = Cell.EMPTY

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
    def astar_helper(self, start: Cell, end: Cell, g: list, h: list, open_list: BinHeap, visited: set, visited_blocked: set, search: list, counter: int) -> List[List[Optional[State]]]:
        tree: List[List[Optional[Cell]]] = [[None for i in range(len(self.world))] for j in range(len(self.world))]
        while len(open_list) > 0 and g[end.row][end.col] > open_list.peek().get_f_value():
            # Remove a state s with the smallest f-value
            s = open_list.pop_root()
            visited.add(s)
            # Explore neighbors
            neighbors = self.get_unvisited_neighbors(s.cell, visited_blocked)
            for neighbor in neighbors:
                if search[neighbor.row][neighbor.col] < counter:
                    g[neighbor.row][neighbor.col] = np.Infinity
                    search[neighbor.row][neighbor.col] = counter
                if g[neighbor.row][neighbor.col] > s.g_value + 1:
                    g[neighbor.row][neighbor.col] = s.g_value + 1
                    tree[neighbor.row][neighbor.col] = s
                    idx = open_list.index_of(neighbor)
                    if idx > 0:
                        open_list.remove(idx)
                    open_list.insert(State(neighbor, g[neighbor.row][neighbor.col], h[neighbor.row][neighbor.col]))
        return tree

    def forward_astar(self, start: Cell, end: Cell, tie_break_smaller_g: bool, adaptive: bool):
        counter = 0
        search: List[List[int]] = [[0 for i in range(len(self.world))] for j in range(len(self.world))]
        g: List[List[Optional[int]]] = [[None for i in range(len(self.world))] for j in range(len(self.world))]
        h: List[List[int]] = [[World.mhd(j, end.row, i, end.col) for i in range(len(self.world))] for j in range(len(self.world))]
        visited_blocked = set()
        while not start.__eq__(end):
            counter += 1
            g[start.row][start.col] = 0
            search[start.row][start.col] = counter
            g[end.row][end.col] = np.Infinity
            search[end.row][end.col] = counter
            open_list = BinHeap(tie_break_smaller_g)
            open_list.insert(State(start, g[start.row][start.col], h[start.row][start.col]))
            visited = set()

            for direction in Agent.DIRECTIONS:
                neighbor = self.move(start, direction)
                if neighbor is not None and neighbor.type == Cell.OBSTACLE:
                    visited_blocked.add(neighbor)
            tree = self.astar_helper(start, end, g, h, open_list, visited, visited_blocked, search, counter)

            if len(open_list) == 0:
                print('I cannot reach the target ;-;')
                return

            # follow tree pointers from goal state to start state, then move agent along resulting path from start state to goal state until it reaches goal state
            #       or one or more action costs on the path increase;
            path: List[State] = list()
            path.insert(0, State(end, g[end.row][end.col], h[end.row][end.col]))
            while path[0].cell.row != start.row or path[0].cell.col != start.col:
                path.insert(0, tree[path[0].cell.row][path[0].cell.col])

            for item in path[1:]:
                if item.cell.type == Cell.OBSTACLE:
                    visited_blocked.add(item.cell)
                    # print('\t\t obstacle:', item.cell.row, item.cell.col)
                    break
                else:
                    self.world.grid[item.cell.row][item.cell.col].type = Cell.PATH
                    start = item.cell

            if adaptive:
                for item in visited:
                    h[item.cell.row][item.cell.col] = g[end.row][end.col] - g[item.cell.row][item.cell.col]

        print('I reached the target! :D')

    def backward_astar(self, end: Cell, start: Cell, tie_break_smaller_g: bool, adaptive: bool):
        counter = 0
        search: List[List[int]] = [[0 for i in range(len(self.world))] for j in range(len(self.world))]
        g: List[List[Optional[int]]] = [[None for i in range(len(self.world))] for j in range(len(self.world))]
        h: List[List[int]] = [[World.mhd(j, end.row, i, end.col) for i in range(len(self.world))] for j in range(len(self.world))]
        visited_blocked = set()
        while not start.__eq__(end):
            counter += 1
            g[start.row][start.col] = 0
            search[start.row][start.col] = counter
            g[end.row][end.col] = np.Infinity
            search[end.row][end.col] = counter
            open_list = BinHeap(tie_break_smaller_g)
            open_list.insert(State(start, g[start.row][start.col], h[start.row][start.col]))
            visited = set()

            for direction in Agent.DIRECTIONS:
                neighbor = self.move(start, direction)
                if neighbor is not None and neighbor.type == Cell.OBSTACLE:
                    visited_blocked.add(neighbor)
            tree = self.astar_helper(start, end, g, h, open_list, visited, visited_blocked, search, counter)

            if len(open_list) == 0:
                print('I cannot reach the target ;-;')
                return

            # follow tree pointers from goal state to start state, then move agent along resulting path from start state to goal state until it reaches goal state
            #       or one or more action costs on the path increase;
            path: List[State] = list()
            path.append(State(end, g[end.row][end.col], h[end.row][end.col]))
            while path[len(path) - 1].cell.row != start.row or path[len(path) - 1].cell.col != start.col:
                path.append(tree[path[len(path) - 1].cell.row][path[len(path) - 1].cell.col])

            for item in path[1:]:
                if item.cell.type == Cell.OBSTACLE:
                    visited_blocked.add(item.cell)
                    # print('\t\t obstacle:', item.cell.row, item.cell.col)
                    break
                else:
                    self.world.grid[item.cell.row][item.cell.col].type = Cell.PATH
                    end = item.cell

            if adaptive:
                for item in visited:
                    h[item.cell.row][item.cell.col] = g[end.row][end.col] - g[item.cell.row][item.cell.col]

        print('I reached the target! :D')


def main():
    # myWorld = World(25)
    # # myWorld = World(0, [['█', '█', '█', '█', '█'], ['█', '█', '█', '█', '█'], ['█', '█', '*', '█', '█'], ['█', '█', '*', '█', '█'], ['█', '█', '█', '*', '█']])
    # myAgent = Agent(myWorld)
    # myAgent.draw_obstacles()
    #
    # # obstacles = 0
    # # emptys = 0
    # # nulls = 0
    # # for cell in myWorld.grid.flatten():
    # #     if cell.type == Cell.EMPTY:
    # #         emptys += 1
    # #     if cell.type == Cell.OBSTACLE:
    # #         obstacles += 1
    # #     if cell.type == Cell.NULL:
    # #         nulls += 1
    # # print('emptys: ', emptys / (emptys + obstacles + nulls), ' obstacles: ', obstacles / (emptys + obstacles + nulls), ' nulls: ', nulls / (emptys + obstacles + nulls))
    #
    # emptys = [cell for cell in myAgent.world.grid.flatten() if cell.type == Cell.EMPTY]
    # start, end = np.random.choice(emptys, 2, replace=False)
    # # start = myAgent.world.grid[4][2]
    # # end = myAgent.world.grid[4][4]
    #
    # myAgent.print_world(start, end)
    # myAgent.forward_astar(start, end, tie_break_smaller_g=False, adaptive=False)
    # myAgent.print_world(start, end)
    # # myAgent.reset()
    # # myAgent.forward_astar(start, end, tie_break_smaller_g=False, adaptive=False)
    # # myAgent.print_world(start, end)
    # myAgent.reset()
    # myAgent.forward_astar(start, end, tie_break_smaller_g=False, adaptive=True)
    # myAgent.print_world(start, end)

    forward_total_time_smaller_g = 0
    forward_total_time_larger_g = 0
    adaptive_total_time = 0
    backward_total_time = 0
    runs = 50
    size = 101
    for i in range(runs):
        my_agent = Agent(World(size))
        my_agent.draw_obstacles()
        emptys = [cell for cell in my_agent.world.grid.flatten() if cell.type == Cell.EMPTY]
        start, end = np.random.choice(emptys, 2, replace=False)

        start_time = dt.datetime.now()
        my_agent.forward_astar(start, end, tie_break_smaller_g=False, adaptive=False)
        forward_total_time_larger_g += (dt.datetime.now() - start_time).microseconds

        my_agent.reset()
        start_time = dt.datetime.now()
        my_agent.forward_astar(start, end, tie_break_smaller_g=True, adaptive=False)
        forward_total_time_smaller_g += (dt.datetime.now() - start_time).microseconds

        my_agent.reset()
        start_time = dt.datetime.now()
        my_agent.forward_astar(start, end, tie_break_smaller_g=False, adaptive=True)
        adaptive_total_time += (dt.datetime.now() - start_time).microseconds

        my_agent.reset()
        start_time = dt.datetime.now()
        my_agent.backward_astar(start, end, tie_break_smaller_g=False, adaptive=False)
        backward_total_time += (dt.datetime.now() - start_time).microseconds

    print('Forward A* Tie break favor larger g total time: ', forward_total_time_larger_g / math.pow(10, 6), 's')
    print('Forward A* Tie break favor smaller g total time: ', forward_total_time_smaller_g / math.pow(10, 6), 's')
    print('Backward A* Tie break favor larger g total time: ', backward_total_time / math.pow(10, 6), 's')
    print('Adaptive A* total time: ', adaptive_total_time / math.pow(10, 6), 's')


if __name__ == '__main__':
    main()
