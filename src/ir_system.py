import re
from collections import defaultdict
import math


class IRSystem():
    def __init__(self, dataset):
        self.corpus = dataset()
        self.index = self.make_index()
        self.N = len(self.corpus)

    def df(self, term):
        return self.index[term].df()

    def idf(self, term):
        return self.index[term].idf(self.N)
    
    def tf_idf(self, term, docid):
        return self.index[term].tf_idf(docid, self.N)
    
    def check_posting(self, term, docid):
        return self.index[term].check_posting(docid)
    
    def postings(self, term, key=None):
        return self.index[term].postings(key)

    def make_index(self):
        """this function create a index whith tf-idf"""
        terms = defaultdict(Posting_list)
        for docid, article in enumerate(self.corpus):
            for term in article:
                if term not in terms:
                    terms[term] = Posting_list()
                terms[term].add_posting(docid)
        return terms





class Posting_list():
    def __init__(self):
        self._postings = {}
        self._df = 0

    def add_posting(self, docid):
        if docid not in self._postings:
            self._postings[docid] = 1
 # type: ignore            
            self._df += 1
        else:
            self._postings[docid] += 1


    def check_posting(self, docid):
        return docid in self._postings

    def postings(self, key=None):
        if key in self._postings:
            return self._postings[key]
        elif key is not None:
            print("Key not found")
        else:
           return self._postings

    def df(self):
        return self._df
    
    def idf(self, N):
        return math.log(N/self._df)
    
    def tf_idf(self, docid, N):
        return self._postings[docid]*self.idf(N)



    