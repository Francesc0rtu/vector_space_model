import re

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
    def __getitem__(self, key):
        if key == "docid":
            return self.docid
        elif key == "score":
            return self.score
        else:
            raise KeyError

class Heap():
    def __init__(self):
        self.heap = [] 
        self.docid_to_index = {}
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return str(self.heap)
    def __len__(self):
        return len(self.heap)
    def __getitem__(self, key):
        return self.heap[key]

    def add(self, docid, score):
        self.heap.append(Heap_element(docid, score))
        self.docid_to_index[docid] = len(self.heap)-1
        index = self.heapify_up(len(self.heap)-1)
        self.docid_to_index[docid] = index
        

    def pop(self):
        if len(self.heap) == 0:
            return None
        self.swap(0, len(self.heap)-1)
        element = self.heap.pop()
        self.heapify_down(0)
        del self.docid_to_index[element.docid]

        return element
    
    def heapify_up(self, index):
        if index == 0:
            return 0
        parent = (index-1)//2
        if self.heap[parent] < self.heap[index]:
            self.swap(parent, index)
            self.docid_to_index[self.heap[parent].docid] , self.docid_to_index[self.heap[index].docid] = self.docid_to_index[self.heap[index].docid], self.docid_to_index[self.heap[parent].docid]
            index = self.heapify_up(parent)
        return index
    
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
            self.docid_to_index[self.heap[index].docid] , self.docid_to_index[self.heap[largest].docid] = self.docid_to_index[self.heap[largest].docid], self.docid_to_index[self.heap[index].docid]
            index = self.heapify_down(largest)
        return index
    
            
    def swap(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

    def top(self):
        return self.heap[0]
    
    def empty(self):
        return len(self.heap) == 0
    
    def contains(self, docid):
        return docid in self.docid_to_index
    
    def update(self, docid, score):
        if docid not in self.docid_to_index:
            self.add(docid, score)
            return
        index = self.docid_to_index[docid]
        self.heap[index].score = score
        index = self.heapify_up(index)
        self.docid_to_index[docid] = index

    
    def get_top_k(self, k):
        """get top k elements from heap"""
        return self.heap[0:k]
    
    def verify_heap_proprerty(self):
        """verify heap property"""
        for i in range(len(self.heap)):
            left = 2*i+1
            right = 2*i+2
            if left < len(self.heap) and self.heap[left] > self.heap[i]:
                print("left child is greater than parent")
                return False
            if right < len(self.heap) and self.heap[right] > self.heap[i]:
                print("right child is greater than parent")
                return False
        return True
    

def preprocess_query(query):
    """preprocess query"""
    query = query.lower()
    query = re.sub(r'[^a-zA-Z\s]+', '', query)
    query = query.split()
    return query
    
    
        
def preprocess_row(row):
    """preprocess row"""
    row = row.lower()
    row = re.sub(r'[^a-zA-Z\s]+', '', row)
    row = row.split()
    return row