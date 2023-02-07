import re
from collections import defaultdict, OrderedDict
import math
from src.utils import Heap
from tqdm import tqdm


class IRSystem():
    
    def __init__(self, dataset):
        self.corpus = dataset()
        self.index = Index(self.corpus)
        self.N = len(self.corpus)

    def search(self, query):
        """This function return the top 10 documents of a query"""
        


class Posting_list():
    def __init__(self):
        self._postings = []
        self._df = 0

    def __repr__(self):
        return str(self._postings)

    def __iter__(self):
        return iter(self._postings)

    def add_posting(self, docid):
        index = self.find_posting_index(docid)
        self._df += 1
        if index is not None:
            self._postings[index] = (docid, self._postings[index][1] + 1)
        else:
            index = len(self._postings)
            self._postings.append((docid, 1))

        self.keep_postings_sorted(index)
    
    def keep_postings_sorted(self, index):
        while index > 0 and self._postings[index][1] > self._postings[index-1][1]:
            self._postings[index], self._postings[index-1] = self._postings[index-1], self._postings[index]
            index -= 1

    def find_posting_index(self, docid):
        for index, posting in enumerate(self._postings):
            if posting[0] == docid:
                return index
        return None
    

class Index():
    def __init__(self, corpus):
        self._terms = {}
        self.make_index(corpus)
    
    def __repr__(self):
        return str(self._terms)
    
    def __getitem__(self, term):
        try:
            return self._terms[term]
        except KeyError:
            return None

    def __contains__(self, term):
        try:
            self._terms[term]
            return True
        except KeyError:
            return False

    def df(self, term):
        return self._terms[term]._df
    
    def idf(self, term, N):
        return math.log(N/self.df(term))
    
    def tf(self, term, docid):
        return self._terms[term].find_posting_index(docid).tf()
    
    def tf_idf(self, term, docid, N):
        return self.tf(term, docid)*self.idf(term, N)
        
    def make_index(self, corpus):
        for docid, doc in tqdm(enumerate(corpus), total=len(corpus)):
            for term in doc:
                try:
                    self._terms[term].add_posting(docid)
                except KeyError:
                    self._terms[term] = Posting_list()
                    self._terms[term].add_posting(docid)
    

 