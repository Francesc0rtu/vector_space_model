import re
from collections import defaultdict, OrderedDict
import math
from src.utils import Heap, preprocess_row
from src.dataloader import Dataset
from tqdm import tqdm
import numpy as np
import pickle
import os



class IRSystem():
    
    def __init__(self, dataset="time", load=False):
        if load:  # load model 
            if dataset == "time":
                path = "MODEL/time"
                if os.path.exists(path):
                    self.index = pickle.load(open(path + "/index.pkl", "rb"))
                    self.corpus = pickle.load(open(path + "/corpus.pkl", "rb"))
                    self.query = pickle.load(open(path + "/query.pkl", "rb"))
                    self.N = len(self.corpus)
                else:
                    raise FileNotFoundError("No model found")
        else:    # create model
            if dataset == "time":
                self.dataset = Dataset(dataset = "time")
                self.corpus = self.dataset['corpus']
                self.query = self.dataset['query']

            
            self.index = Index(self.corpus) # create index
            self.index.remove_stopwords() # remove stopwords
            self.N = len(self.corpus)

    def __getitem__(self, key):
        if key == 'index':
            return self.index
        elif key == 'corpus':
            return self.corpus
        elif key == 'query':
            return self.query
        else:
            raise KeyError

    def get_query_vector(self, query):
        """This function return the vector of a query"""
        query = preprocess_row(query)
        query_vector = np.zeros(len(self.index))
        for term in query:
            if term in self.index:
                query_vector[self.index[term].position] = 1

        # normalize
        query_vector = query_vector/np.linalg.norm(query_vector)
        return query_vector
    
    def get_doc_vectors(self):
        """ This function return the vector of each document"""
        doc_vectors = []
        for docid in range(self.N):
            doc_vectors.append(self.get_doc_vector(docid))
        return doc_vectors

    def get_doc_vector(self, docid):
        """This function return the vector of a document"""

        doc_vector = np.zeros(len(self.index))
        for term in self.corpus[docid]:
            if term in self.index:
                doc_vector[self.index[term].position] = self.index.tf_idf(term, docid, self.N)
        # normalize
        doc_vector = doc_vector/np.linalg.norm(doc_vector)            
        return doc_vector

    
    def get_docs_from_query(self, query, champions=False):
        """ This function compute the vector of each document and the query vector symultaneously while iterating over the posting lists,
        then compute the cosine similarity and return the heap of the top documents"""
        if type(query) == str:
            query = preprocess_row(query)
        elif type(query) != list:
            raise TypeError("query must be a list or a string")

        vectors = {}
        query_vector = np.zeros(len(self.index))
        heap = Heap()
        for term in query:
            if term in self.index:
                print(term)
                query_vector[self.index[term].position()] = 1
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector/np.linalg.norm(query_vector)
                
                champ_iter = 0
                for docid, _ in self.index[term]:
                    
                    if champ_iter > (self.N/2) and champions == True:
                        break
                    champ_iter += 1

                    if docid in vectors:
                        vectors[docid][self.index[term].position()] = self.index.tf_idf(term, docid, self.N)
                    else:
                        vectors[docid] = np.zeros(len(self.index))
                        vectors[docid][self.index[term].position()] = self.index.tf_idf(term, docid, self.N)

                    norm = np.linalg.norm(vectors[docid])
                    if norm > 0:
                        vectors[docid] = vectors[docid]/np.linalg.norm(vectors[docid])
                    score = np.dot(vectors[docid], query_vector)
                    heap.update(docid, score)
            elif term not in self.index:
                print("The term {} is not in the index".format(term))
       
        return vectors, heap
    
                        
    def cosine_similiarity(self, query_vector, doc_vector):
        return np.dot(query_vector, doc_vector)
    
    def search(self, query, k=10):
        """ This function return the top k documents of a query
        """
        vectors, heap = self.get_docs_from_query(query)
        if k==None:
            return heap
        else:
            return heap.get_top_k(k)
    
    def get_document_from_list(self, heap):
        """ This function return the document from a list of heap-elements"""
        documents = []
        for heap_el in heap:
            
            documents.append(self.corpus[heap_el["docid"]])
        
        return documents
    


class Posting_list():
    def __init__(self, position):
        self._postings = []
        self._df = 0
        self._position = position

    def __repr__(self):
        return str(self._postings)

    def __iter__(self):
        return iter(self._postings)
    

    def add_posting(self, docid):
        index = self.find_posting_index(docid)
        if index is not None:
            self._postings[index] = (docid, self._postings[index][1] + 1)
        else:
            self._df += 1
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

    def position(self):
        return self._position
    
    def get_post(self, docid):
        index = self.find_posting_index(docid)
        if index is not None:
            return self._postings[index]
        else:
            return None
  


class Index():
    def __init__(self, corpus):
        self._terms = OrderedDict()
        self.make_index(corpus)
        self.number_of_documents = len(corpus)
    
    def __repr__(self):
        return str(self._terms)
    
    def __getitem__(self, term):
            return self._terms[term]


    def __contains__(self, term):
        try:
            self._terms[term]
            return True
        except KeyError:
            return False
        
    def __len__(self):
        return len(self._terms)
    
    def __iter__(self):
        return iter(self._terms)
    

    def df(self, term):
        return self._terms[term]._df
    
    def idf(self, term):
        return math.log10(self.number_of_documents/self.df(term))
    
    def tf(self, term, docid):
        post = self._terms[term].get_post(docid)
        if post is not None:
            return post[1]
        else:
            return 0
    
    def tf_idf(self, term, docid, N):
        return self.tf(term, docid)*self.idf(term)
        
    def make_index(self, corpus):
        position = 0
        for docid, doc in tqdm(enumerate(corpus), total=len(corpus), desc="Indexing:"):
            for term in doc:
                try:
                    self._terms[term].add_posting(docid)
                except KeyError:
                    self._terms[term] = Posting_list(position)
                    position += 1
                    self._terms[term].add_posting(docid)
    
    def remove_stopwords(self, percentage = 0.05):
        """ This function remove the terms that appear in more than percentage of the documents
        """
        for term in list(self._terms):
            if self.df(term) > percentage*self.number_of_documents:
                del self._terms[term]

 