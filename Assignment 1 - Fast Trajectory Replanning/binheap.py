import numpy as np

class BinHeap:

    #Basic helper methods
    def __init__(self,capacity):
        self.capacity = capacity
        self.arr = np.zeros(capacity,dtype=int)
        self.size = 0
    
    def get_parent_index(self,index):
        return (index-1)//2
    def get_left_child_index(self,index):
        return (index*2)+1
    def get_right_child_index(self,index):
        return (index*2)+2

    def has_parent(self,index):
        return self.get_parent_index(index) >= 0
    def has_left_child(self,index):
        return self.get_left_child_index(index) < self.size
    def has_right_child(self,index):
        return self.get_right_child_index(index) < self.size

    def parent(self,index):
        return self.arr[self.get_parent_index(index)]
    def left_child(self,index):
        return self.arr[self.get_left_child_index(index)]
    def right_child(self,index):
        return self.arr[self.get_right_child_index(index)]
    
    def swap(self, index1, index2):
        temp = self.arr[index1]
        self.arr[index1]=self.arr[index2]
        self.arr[index2] = temp

    def print_heap(self):
        print(self.arr[:self.size])

    #functional methods
    def heapify_up(self,index):
        if (self.has_parent(index) and (self.parent(index) > self.arr[index])):
            self.swap(self.get_parent_index(index),index)
            index = self.get_parent_index(index)
            self.heapify_up(self.get_parent_index(index))

    def heapify_down(self,index):
        smol_child_index = index

        if (self.has_left_child(index) and self.arr[smol_child_index] > self.left_child(index)):
            smol_child_index = self.get_left_child_index(index)

        if(self.has_right_child(index) and self.arr[smol_child_index] > self.right_child(index)):
            smol_child_index = self.get_right_child_index(index)

        if smol_child_index != index:
            self.swap(index,smol_child_index)
            self.heapify_down(smol_child_index)

    def insert(self,data):
        if(self.size==self.capacity):
            print('heap is full, allocating more space')
            self.arr = np.resize(self.arr,new_shape = [self.capacity*2])
            self.capacity *=2
        self.arr[self.size] = data
        self.size+=1
        self.heapify_up(self.size-1)

    def pop_root(self):
        if (self.size==0):
            print("Empty heap")
            return
        data = self.arr[0]
        self.arr[0] = self.arr[self.size-1]
        self.size-=1
        self.heapify_down(0)
        return data

    


def main():
    openlist = BinHeap(1)
    openlist.insert(10)
    openlist.print_heap()
    openlist.insert(5)
    openlist.print_heap()
    openlist.insert(3)
    openlist.print_heap()
    openlist.insert(7)
    openlist.print_heap()
    openlist.insert(12)
    openlist.print_heap()
    
    while openlist.size!=0:
        min_val = openlist.pop_root()
        print(min_val,openlist.arr[:openlist.size])

if __name__ == '__main__':
    main()