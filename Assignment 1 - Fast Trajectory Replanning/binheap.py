from typing import List, Optional


class Cell:
    NULL = '-'
    EMPTY = '█'
    OBSTACLE = '*'
    PLAYER = 'P'
    TARGET = '$'

    # def can_move_to(self):
    #     return self.type == Cell.EMPTY # or self.type == Cell.TARGET

    def __init__(self, row, col, cell_type=NULL):
        self.type = cell_type
        self.row = row
        self.col = col

    def __str__(self):
        return self.type

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __hash__(self):
        return id(self)


class State:

    def __init__(self, cell: Cell, g_value: int, h_value: int):
        self.cell = cell
        self.g_value = g_value
        self.h_value = h_value

    def get_f_value(self):
        return self.g_value + self.h_value

    def __str__(self):
        return '(' + str(self.cell.row) + ', ' + str(self.cell.col) + '): ' + str(self.get_f_value())
        # return str(self.f_value)

    def cmp(self, other, tie_break_smaller_g: bool):
        f = self.get_f_value() - other.get_f_value()
        if f == 0:
            if tie_break_smaller_g:
                return other.g_value - self.g_value
            else:
                return self.g_value - other.g_value
        return f


class BinHeap:

    # Basic helper methods
    def __init__(self, tie_break_smaller_g: bool = True):
        self.arr: List[State] = list()
        self.tie_break_smaller_g = tie_break_smaller_g

    @staticmethod
    def get_parent_index(index: int) -> int:
        return (index - 1) // 2

    @staticmethod
    def get_left_child_index(index: int) -> int:
        return (index * 2) + 1

    @staticmethod
    def get_right_child_index(index: int) -> int:
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
        if self.has_parent(index) and (self.parent(index).cmp(self.arr[index], self.tie_break_smaller_g) > 0):
            parent = self.get_parent_index(index)
            self.swap(parent, index)
            self.heapify_up(parent)

    def heapify_down(self, index) -> None:
        smol_child_index = index

        if self.has_left_child(index) and self.arr[smol_child_index].cmp(self.left_child(index), self.tie_break_smaller_g) > 0:
            smol_child_index = self.get_left_child_index(index)

        if self.has_right_child(index) and self.arr[smol_child_index].cmp(self.right_child(index), self.tie_break_smaller_g) > 0:
            smol_child_index = self.get_right_child_index(index)

        if smol_child_index != index:
            self.swap(index, smol_child_index)
            self.heapify_down(smol_child_index)

    def insert(self, state: State) -> None:
        self.arr.append(state)
        self.heapify_up(len(self.arr) - 1)

    def pop_root(self) -> Optional[State]:
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

    def remove(self, idx: int) -> None:
        self.arr.pop(idx)

    def __len__(self):
        return len(self.arr)


def main():
    openlist = BinHeap()
    openlist.insert(State(Cell(0, 0), 1, 0))
    # openlist.print_heap()
    openlist.insert(State(Cell(1, 1), 5, 0))
    # openlist.print_heap()
    openlist.insert(State(Cell(2, 2), 12, 0))
    # openlist.print_heap()
    openlist.insert(State(Cell(3, 3), -5, 0))
    # openlist.print_heap()
    openlist.insert(State(Cell(4, 4), 3, 0))
    openlist.print_heap()

    print('popping heap')
    while len(openlist) > 0:
        min_val = openlist.pop_root()
        print(min_val)


if __name__ == '__main__':
    main()
