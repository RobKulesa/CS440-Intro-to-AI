from typing import Union, List

import numpy as np

from main import Cell


class State:
    def __init__(self, cell: Cell, f_value: int):
        self.cell = cell
        self.f_value = f_value

    def __str__(self):
        # return '(' + str(self.cell.row) + ', ' + str(self.cell.col) + '): ' + str(self.f_value)
        return str(self.f_value)

    def __cmp__(self, other):
        return self.f_value - other.f_value


class BinHeap:

    # Basic helper methods
    def __init__(self):
        self.arr: List[State] = list()

    def get_parent_index(self, index: int) -> int:
        return (index - 1) // 2

    def get_left_child_index(self, index: int) -> int:
        return (index * 2) + 1

    def get_right_child_index(self, index: int) -> int:
        return (index * 2) + 2

    def has_parent(self, index: int) -> bool:
        return self.get_parent_index(index) >= 0

    def has_left_child(self, index: int) -> bool:
        return self.get_left_child_index(index) < len(self.arr)

    def has_right_child(self, index: int) -> bool:
        return self.get_right_child_index(index) < len(self.arr)

    def parent(self, index: int) -> State:
        return self.arr[self.get_parent_index(index)]

    def left_child(self, index: int) -> State:
        return self.arr[self.get_left_child_index(index)]

    def right_child(self, index: int) -> State:
        return self.arr[self.get_right_child_index(index)]

    def swap(self, index1: int, index2: int) -> None:
        temp = self.arr[index1]
        self.arr[index1] = self.arr[index2]
        self.arr[index2] = temp

    def print_heap(self) -> None:
        print('heap: ', *self.arr)

    # functional methods
    def heapify_up(self, index: int) -> None:
        if self.has_parent(index) and (self.parent(index).__cmp__(self.arr[index]) > 0):
            parent = self.get_parent_index(index)
            self.swap(parent, index)
            self.heapify_up(parent)

    def heapify_down(self, index) -> None:
        smol_child_index = index

        if self.has_left_child(index) and self.arr[smol_child_index].__cmp__(self.left_child(index)) > 0:
            smol_child_index = self.get_left_child_index(index)

        if self.has_right_child(index) and self.arr[smol_child_index].__cmp__(self.right_child(index)) > 0:
            smol_child_index = self.get_right_child_index(index)

        if smol_child_index != index:
            self.swap(index, smol_child_index)
            self.heapify_down(smol_child_index)

    def insert(self, state: State) -> None:
        self.arr.append(state)
        self.heapify_up(len(self.arr) - 1)

    def pop_root(self) -> State:
        if len(self) == 0:
            print("Empty heap")
            return
        state = self.arr.pop(0)
        self.heapify_down(0)
        return state

    def peek(self) -> State:
        return self.arr[0]

    def get_cells(self) -> List[Cell]:
        return [i.cell for i in self.arr]

    def index_of(self, cell: Cell) -> int:
        for i, e in enumerate(self.arr):
            if e is State and e.cell.row == cell.row and e.cell.col == cell.col:
                return i
        return -1

    def __len__(self):
        return len(self.arr)


def main():
    openlist = BinHeap()
    openlist.insert(State(Cell(0, 0), 1))
    # openlist.print_heap()
    openlist.insert(State(Cell(1, 1), 5))
    # openlist.print_heap()
    openlist.insert(State(Cell(2, 2), 12))
    # openlist.print_heap()
    openlist.insert(State(Cell(3, 3), -5))
    # openlist.print_heap()
    openlist.insert(State(Cell(4, 4), 3))
    openlist.print_heap()

    print('popping heap')
    while len(openlist) > 0:
        min_val = openlist.pop_root()
        print(min_val)


if __name__ == '__main__':
    main()
