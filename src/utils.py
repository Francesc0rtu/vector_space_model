class Heap_element():
    def __init__(self, docid, score):
        self.docid = docid
        self.score = score
    def __lt__(self, other):
        return self.score < other.score
    def __gt__(self, other):
        return self.score > other.score
    def __eq__(self, other):
        return self.score == other.score
    def __repr__(self):
        return "docid:"+ str(self.docid) + " score:" + str(self.score)

class Heap():
    def __init__(self):
        self.heap = []
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return str(self.heap)
    def __len__(self):
        return len(self.heap)
    def __getitem__(self, key):
        return self.heap[key]
    # def __setitem__(self, key, value):
    #     self.heap[key] = value
    def add(self, docid, score):
        self.heap.append(Heap_element(docid, score))
        self.heapify_up(len(self.heap)-1)
    def pop(self):
        if len(self.heap) == 0:
            return None
        self.swap(0, len(self.heap)-1)
        element = self.heap.pop()
        self.heapify_down(0)
        return element
    def heapify_up(self, index):
        if index == 0:
            return
        parent = (index-1)//2
        if self.heap[parent] < self.heap[index]:
            self.swap(parent, index)
            self.heapify_up(parent)
    def heapify_down(self, index):
        left = 2*index+1
        right = 2*index+2
        largest = index
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        if largest != index:
            self.swap(index, largest)
            self.heapify_down(largest)
    def swap(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]
    def top(self):
        return self.heap[0]
    def empty(self):
        return len(self.heap) == 0
    def get_best_k(self, k):
        best_k = []
        for i in range(k):
            best_k.append(self.pop())
        return best_k
    def contains(self, docid):
        for element in self.heap:
            if element.docid == docid:
                return True
        return False
    def update(self, docid, score):
        for element in self.heap:
            if element.docid == docid:
                element.score = score
                self.heapify_up(self.heap.index(element))
                return
        self.add(docid, score)
        