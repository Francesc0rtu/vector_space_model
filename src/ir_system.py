import re
from collections import defaultdict, OrderedDict
import math
from src.utils import Heap, preprocess_row
from src.dataloader import Dataset
from tqdm import tqdm
import numpy as np
import pickle
import os
from typing import Any



class IRSystem():
    """ This class represent the information retrieval system"""
    
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
                self.dataset = Dataset(dataset = "time") # load dataset
                self.corpus = self.dataset['corpus'] # get corpus
                self.query = self.dataset['query'] # get query

            
            self.index = Index(self.corpus) # create index
            self.index.remove_stopwords() # remove stopwords
            self.N = len(self.corpus) # number of documents in the corpus

    def __getitem__(self, key: str) -> Any: 
        """ This function return the index, the corpus or the query"""
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
        query = preprocess_row(query) # preprocess query, remove punctuation
        query_vector = np.zeros(len(self.index)) # create vector of zeros
        for term in query: # for each term in the query
            if term in self.index: # if the term is in the index
                query_vector[self.index[term].position] = 1 # set the value of the vector to 1

        
        query_vector = query_vector/np.linalg.norm(query_vector) # normalize the vector
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
        then compute the cosine similarity and return the heap of the top documents. The main idea is to avoid to compute the vector of each document, so we use the following osservation:
        if a term is not in the query, the entry corrispoding to the term do not contribute to the socre. 
        So we can compute only the document vectors of documents that contains at least one term of the query.
        For this reason we iterate over the term of the query, then we iterate over the posting list of the term and we compute the vector of the document.
        If a document is already in the dictionary of vectors, because it contains another previous term of the query, we just update the entry corresponding to the new term and
        recompute the score. 
        """


        if type(query) == str: # if the query is a string
            query = preprocess_row(query) # preprocess query, remove punctuation, get list of terms
        elif type(query) != list: # if the query is not a string or a list
            raise TypeError("query must be a list or a string")

        vectors = {} # dictionary of vectors of documents
        # initialize vectors 
        query_vector = np.zeros(len(self.index)) 
        tmp_query_vector = np.zeros(len(self.index))
        tmp_vector_docid = np.zeros(len(self.index))

        # initialize heap of top documents
        heap = Heap() 
        
        for term in query: # for each term in the query
            if term in self.index:  # if the term is in the index

                # set the entry corresponding to the term in the query vector to 1
                query_vector[self.index[term].position()] = 1 

                # compute the norm of the query vector
                norm = np.linalg.norm(query_vector) 

                if norm > 0: # if the norm is greater than 0
                    # a tmp vector is created from the query vector normalized
                    tmp_query_vector = query_vector/np.linalg.norm(query_vector)
                
                champ_iter = 0
                for docid, _ in self.index[term]: # for each document in the posting list of the term
                    
                    if champ_iter > (self.N/2) and champions == True:
                        break
                    champ_iter += 1

                    if docid in vectors: # if the vector of the document already exists
                        # set the entry corresponding to the term in the vector of the document to the tf-idf value
                        vectors[docid][self.index[term].position()] = self.index.tf_idf(term, docid, self.N)
                    else: # if the vector of the document does not exist
                        # create a vector of zeros
                        vectors[docid] = np.zeros(len(self.index))
                        # set the entry corresponding to the term in the vector of the document to the tf-idf value
                        vectors[docid][self.index[term].position()] = self.index.tf_idf(term, docid, self.N)

                    # compute the norm of the vector of the document
                    norm = np.linalg.norm(vectors[docid])

                    if norm > 0: # if the norm is greater than 0
                        # a tmp vector is created from the vector of the document normalized
                        tmp_vector_docid = vectors[docid]/np.linalg.norm(vectors[docid])

                    # compute the cosine similarity between the query vector and the vector of the document
                    score = np.dot(tmp_vector_docid, tmp_query_vector)

                    # add the document to the heap, update the score if the document is already in the heap
                    heap.update(docid, score)
       
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
    """ This class represent a posting list of a term"""
    def __init__(self, position):
        self._postings = [] # list of tuples (docid, tf)
        self._df = 0 # document frequency of the term which is linked to the posting list
        self._position = position # position of the term in the dictionary index. This useful to compute the vector of a document
                                  #  whitout iterating over the dictionary index,
                                  # TODO: maybe there is a more elegant way to save this information

    def __repr__(self):
        return str(self._postings)

    def __iter__(self):
        return iter(self._postings)
    

    def add_posting(self, docid):
        """ This function add a posting to the posting list"""
        index = self.find_posting_index(docid)  # check if the posting is already in the posting list, if yes return the index
        if index is not None: # if the posting is already in the posting list, update the tf
            self._postings[index] = (docid, self._postings[index][1] + 1)
        else: # if the posting is not in the posting list, add it
            self._df += 1 # update the document frequency
            index = len(self._postings) # the index of the new posting is the length of the posting list
            self._postings.append((docid, 1)) # add the posting to the posting list

        self.keep_postings_sorted(index) # keep the posting list sorted by tf
    
    def keep_postings_sorted(self, index):
        """ This function keep the posting list sorted by tf"""
        while index > 0 and self._postings[index][1] > self._postings[index-1][1]: # if the tf of the posting is greater than the tf of the previous posting
            self._postings[index], self._postings[index-1] = self._postings[index-1], self._postings[index] # swap the postings
            index -= 1 # update the index

    def find_posting_index(self, docid):
        """ This function return the index of a posting in the posting list"""
        for i, posting in enumerate(self._postings):
            if posting[0] == docid:
                return i
        return None

    def position(self):
        """ This function return the position of the term in the dictionary index"""
        return self._position
    
    def get_post(self, docid):
        """ This function return the posting of a document"""
        index = self.find_posting_index(docid) # find the index of the posting
        if index is not None: # if the posting is in the posting list
            return self._postings[index] # return the posting
        else: # if the posting is not in the posting list
            return None # return None
  


class Index():
    """ This class represent an index of a corpus"""
    def __init__(self, corpus):
        self._terms = OrderedDict()  # dictionary of terms
        self.make_index(corpus)  # make the index
        self.number_of_documents = len(corpus) # number of documents in the corpus
    
    def __repr__(self):
        return str(self._terms)
    
    def __getitem__(self, term: str) -> Posting_list:
        """ This function return the posting list of a term"""
        return self._terms[term]

    def __contains__(self, term):
        """ This function check if a term is in the index"""
        try: # try to get the term
            self._terms[term]
            return True
        except KeyError: # if the term is not in the index
            return False
        
    def __len__(self):
        return len(self._terms)
    
    def __iter__(self):
        """ This function return an iterator over the terms"""
        return iter(self._terms)
    

    def df(self, term):
        """ This function return the df of a term"""
        return self._terms[term]._df
    
    def idf(self, term: str) -> float:
        """ This function compute the idf of a term"""
        return math.log10(self.number_of_documents/self.df(term))
    
    def tf(self, term, docid):
        """ This function compute the tf of a term in a document"""
        post = self._terms[term].get_post(docid) # get the posting of the term in the document
        if post is not None: # if the term is   in the document
            return post[1] 
        else: # if the term is not in the document
            return 0
    
    def tf_idf(self, term: str, docid, N):
        """ This function compute the tf-idf of a term in a document"""
        return self.tf(term, docid)*self.idf(term)
        
    def make_index(self, corpus):
        """ This function make the index of the corpus"""
        position = 0  # position of the term in t_index
        for docid, doc in tqdm(enumerate(corpus), total=len(corpus), desc="Indexing:"):
            for term in doc:
                try:  # if the term is already in the index
                    self._terms[term].add_posting(docid)  #update the posting list
                except KeyError:  # if the term is not in the index
                    self._terms[term] = Posting_list(position)
                    position += 1   # update the position
                    self._terms[term].add_posting(docid)
    
    def remove_stopwords(self, percentage = 0.95):
        """ This function remove the terms that appear in more than percentage of the documents
        """
        for term in list(self._terms): # iterate over a copy of the keys
            if self.df(term) > percentage*self.number_of_documents: # if the term appear in more than percentage of the documents
                del self._terms[term] # remove the term from the index
        self.update_positions() # update the positions of the terms in the index

    def update_positions(self):
        """ This function update the positions of the terms in the index"""
        for i, (key, value) in enumerate(self._terms.items()): # iterate over the terms
            value._position = i  # update the position of the term

    def get_term(self, index):
        """ This function return the term in the index with the given index"""
        for i, (key, value) in enumerate(self._terms.items()):
            if i == index:
                return key

    def check_position(self):
        """ This function check if the position of the terms in the index are correct"""
        for i, (key, value) in enumerate(self._terms.items()):
            if value.position() != i:
                print("error", i, value.position())
                return False
        return True