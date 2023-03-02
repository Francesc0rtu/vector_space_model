import re
from collections import defaultdict, OrderedDict
import math
from src.utils import  preprocess_row
from src.dataloader import Dataset
from tqdm import tqdm
import numpy as np
import pickle
import os
from typing import Any
import random



class IRSystem():
    """ This class represent the information retrieval system"""  
    def __init__(self, dataset="cisi", file_to_load=None):
        """ This function initialize the IRSystem class.
        
        --- Parameters ---
        dataset: str (default: cisi)
        load: bool (default: False), if True load the index from the file

        --- Attributes ---
        dataset: Dataset
        corpus: list of list of str (list of documents)
        query: list of list of str (list of queries)
        index: Index (inverted index of the corpus)
        N: int, number of documents in the corpus
        doc_vectors: dict, key: docid, value: vector of the document
        rel: dict, key: queryid, value: list of relevant documents
        """
        if file_to_load == None:
            if dataset == "cisi":
                self.dataset = Dataset(dataset = "cisi")   # load dataset
                self.corpus = self.dataset['corpus']       # get corpus
                self.query = self.dataset['query']         # get query
                self.rel = self.dataset['rel']
            
            self.index = Index(self.corpus)                # create index
            self.index.remove_stopwords()                  # remove stopwords
            self.N = len(self.corpus)                      # number of documents in the corpus
            self.compute_doc_vectors()                     # compute the vector of each document
        else:
            self.load(file_to_load)

    def __getitem__(self, key: str) -> Any: 
        """ This function return the index, the corpus or the query.
        
        --- Parameters ---
        key: str, the key of the attribute to return (index, corpus, query)

        --- Returns ---
        the attribute corresponding to the key (index, corpus, query)

        --- Note ---
        This function is used to access index, corpus, queries and ground truth relevance with the [] operator.
        """
        if key == 'index': 
            return self.index
        elif key == 'corpus':
            return self.corpus
        elif key == 'query':
            return self.query
        elif key == "rel":
            return self.rel
        else:
            raise KeyError

    def get_query_vector(self, query):
        """This function return the vector of a query.
        
        --- Parameters ---
        query: str or list of str, the query to vectorize

        --- Returns ---
        query_vector: np.array, the vector of the query
        """
        
        if type(query) == str:                                      # if the query is a string
            query = preprocess_row(query)                           # preprocess query, remove punctuation

        query_vector = np.zeros(len(self.index))                    # create vector of zeros
        for term in query:                                          # for each term in the query
            if term in self.index:                                  # if the term is in the index
                query_vector[self.index[term].position()] = 1       # set the value of the vector to 1

        query_vector = query_vector/np.linalg.norm(query_vector, ord=1) # normalize the vector
        return query_vector
    
    def compute_doc_vectors(self):
        """ This function return the vector of each document.
        
        --- Attributes ---
        doc_vectors: dict, key: docid, value: vector of the document
        """
        self.doc_vectors = {}
        for docid in range(self.N):
            self.doc_vectors[docid] = self.get_doc_vector(docid)

    def get_doc_vector(self, docid):
        """This function return the vector of a document.
        
        --- Parameters ---
        docid: int, the id of the document
        
        --- Returns ---
        doc_vector: np.array, the vector of the document
        """
        doc_vector = np.zeros(len(self.index))
        for term in self.corpus[docid]:
            if term in self.index:
                doc_vector[self.index[term].position()] = self.index.tf_idf(term, docid, self.N)
        # normalize
        doc_vector = doc_vector/np.linalg.norm(doc_vector, ord=1)            
        return doc_vector
                       
    def cosine_similiarity(self, query_vector, doc_vector):
        """ This function return the cosine similarity between a two vectors.
        
        --- Parameters ---
        query_vector: np.array, the vector of the query
        doc_vector: np.array, the vector of the document
        
        --- Returns ---
        cosine_sim: float, the cosine similarity between the two vectors
        """
        return np.dot(query_vector, doc_vector)
    
    def __from_vec_to_words__(self, vec):
        """ This function return the words from a vector.
        
        --- Parameters ---
        vec: np.array, the vector to convert
        
        --- Returns ---
        words: list of str, the words corresponding to the vector
        """
        words = []
        for i in range(len(vec)):
            if vec[i] != 0:
                words.append(self.index.get_term(i))
        return words

    
    def __perform_query_optimized__(self, query, threshold = 0, k = 10):
        """ This function return the k most relevant documents for a query. 
        The function compute the score only between the query and the documents that contains at least one term of the query.
        
        --- Parameters ---
        query: str or list of str, the query to perform
        threshold: int (default: 0), the threshold of tf-idf to consider a term in the query
        k: int (default: 10), the number of documents to return
        
        --- Returns ---
        result: list of tuple, the k most relevant documents for the query (docid, score)
        """
        query_vector = self.get_query_vector(query)
        result = OrderedDict()
        for term in query:
            if term in self.index:
                for docid, tf in self.index[term]:
                    if docid not in result:
                        result[docid] = (self.cosine_similiarity(query_vector, self.doc_vectors[docid]))
                    if tf < threshold:
                        continue
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)

        return result[0:k]
    
    def __perform_query__(self, query_vector, k = 10):
        """ This function return the k most relevant documents for a query. Compute the score between the query and all the documents.

        --- Parameters ---
        query_vector: np.array, the vector of the query
        k: int (default: 10), the number of documents to return

        --- Returns ---
        result: list of tuple, the k most relevant documents for the query (docid, score)
        """
        result = OrderedDict()
        for docid in range(self.N):
            result[docid] = self.cosine_similiarity(query_vector, self.doc_vectors[docid])
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return result[0:k]

    def search(self, query, k = 10):
        """ This function return the k most relevant documents for a query.
        
        --- Parameters ---
        query: str or list of str, the query to perform
        k: int (default: 10), the number of documents to return
        
        --- Returns ---
        result: list of tuple, the k most relevant documents for the query (docid, score)
        """
        return self.__perform_query_optimized__(query, k = k)
    
    # def get_document_from_list(self, heap):
    #     """ This function return the document from a list of heap-elements.
        
    #     --- Parameters ---
    #     heap: list of dict, the list of heap-elements
    #     """
    #     documents = []
    #     for heap_el in heap:           
    #         documents.append(self.corpus[heap_el["docid"]])       
    #     return documents

    def evaluate(self, query_idx):
        """ This function evaluate the system"""
        query = self.query[query_idx]
        rel = self.rel[query_idx]
        docid = self.search(query, len(rel))
        return self.compute_precision(docid, rel)

    def compute_precision(self, list1, list2):
        count = 0
        if len(list1) != len(list2):
            raise Exception("List not same ")
        for el in list1:
            if el in set(list2):
                count += 1

        return count/len(list1)
    
    def evaluate_all(self):
        """ This function evaluate the system on all the query.
        
        --- Return ---
        sprec: the mean precision of the system, float
        
        """
        precision = []
        for qid in self.rel:
            precision.append(self.evaluate(qid))
        sprec = sum(precision)/len(precision)
        return sprec
    
    def rocchio_relevance(self, query_vector, relevant_doc, non_relevant_doc, alpha=1, beta=0.75, gamma=0.15):
        """ This function implement the rocchio relevance feedback algorithm.
        
        --- Parameters ---
        query_vector: the vector of the query, np.array
        relevant_doc: the list of relevant document, list of int
        non_relevant_doc: the list of non relevant document, list of int
        alpha: the weight of the query vector, float
        beta: the weight of the relevant document vector, float
        gamma: the weight of the non relevant document vector, float

        --- Return ---
        new_query_vector: the new query vector, np.array
        """

        relevant_doc_vector = np.zeros(len(self.index))
        non_relevant_doc_vector = np.zeros(len(self.index))

        for docid in relevant_doc:
            relevant_doc_vector += self.doc_vectors[docid]
        relevant_doc_vector = relevant_doc_vector/len(relevant_doc)

        for docid in non_relevant_doc:
            non_relevant_doc_vector += self.doc_vectors[docid]
        non_relevant_doc_vector = non_relevant_doc_vector/len(non_relevant_doc)

        new_query_vector = alpha*query_vector + beta*relevant_doc_vector - gamma*non_relevant_doc_vector
        new_query_vector = new_query_vector/np.linalg.norm(new_query_vector, ord=1)
        return new_query_vector
    
    def pseudo_relevance(self, query, k):
        """ This function implement the pseudo relevance feedback algorithm.
        
        --- Parameters ---
        query: the query to perform, list of string (e.g. ["cat", "dog"])
        k: the number of documents to retrieve

        --- Return ---
        The list of the k most relevant documents, list of docid (e.g. [1, 2, 3])
        """
        k = k * 2
        # Compute the query vector
        query_vec = self.get_query_vector(query)

        # Perform the first query and get the first k documents
        starting_docs = [doc[0] for doc in self.search(query, k)]

        
        # Get the relevant and non relevant documents
        relevant_docs = starting_docs[0:int(k/2)]
        non_relevant_docs = starting_docs[int(k/2):k]

        # Perform the rocchio relevance feedback algorithm k times
        for i in range(k):
            # Compute the new query vector performing the rocchio relevance feedback algorithm
            query_vec = self.rocchio_relevance(query_vec, relevant_docs, non_relevant_docs)
            # Get the new relevant and non relevant documents
            starting_docs = [doc[0] for doc in self.__perform_query__(query_vec, k)]
            relevant_docs = starting_docs[0:int(k/2)]
            non_relevant_docs = starting_docs[int(k/2):k]
        
        # Return the relevant documents
        return relevant_docs
    
    def save(self, filename):

        with open(filename, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            temp_dict = pickle.load(f)
        self.__dict__.update(temp_dict) 
class Posting_list():
    """ This class represent a posting list of a term"""
    def __init__(self, position):
        """ This function initialize the posting list.
        
        --- Parameters ---
        position: int, the position of the term in the dictionary index
        
        --- Attributes ---
        _postings: list of tuple, the list of the posting of the term (docid, tf)
        _df: int, the document frequency of the term
        _position: int, the position of the term in the dictionary index
        """
        self._postings = []                # list of tuples (docid, tf)
        self._df = 0                       # document frequency of the term which is linked to the posting list
        self._position = position          # position of the term in the dictionary index. This useful to compute the vector of a document
                                           #  whitout iterating over the dictionary index,
                                           # TODO: maybe there is a more elegant way to save this information

    def __repr__(self):
        """ This function return the string representation of the posting list"""
        return str(self._postings)

    def __iter__(self):
        """ This function return an iterator over the posting list"""
        return iter(self._postings)
    

    def add_posting(self, docid):
        """ This function add a posting to the posting list.
        
        --- Parameters ---
        docid: int, the id of the document to add to the posting list
        """
        index = self.find_posting_index(docid)      # check if the posting is already in the posting list, if yes return the index
        if index is not None:                       # if the posting is already in the posting list, update the tf
            self._postings[index] = (docid, self._postings[index][1] + 1)
        else:                                       # if the posting is not in the posting list, add it
            self._df += 1                           # update the document frequency
            index = len(self._postings)             # the index of the new posting is the length of the posting list
            self._postings.append((docid, 1))       # add the posting to the posting list

        self.keep_postings_sorted(index)            # keep the posting list sorted by tf
    
    def keep_postings_sorted(self, index):
        """ This function keep the posting list sorted by tf.
        
        --- Parameters ---
        index: int, the index of the posting to check
        """
        while index > 0 and self._postings[index][1] > self._postings[index-1][1]: # if the tf of the posting is greater than the tf of the previous posting
            self._postings[index], self._postings[index-1] = self._postings[index-1], self._postings[index] # swap the postings
            index -= 1  # update the index

    def find_posting_index(self, docid):
        """ This function return the index of a posting in the posting list.
        
        --- Parameters ---
        docid: int, the id of the document to find in the posting list
        
        --- Return ---
        index: int, the index of the posting in the posting list"""
        for i, posting in enumerate(self._postings):
            if posting[0] == docid:
                return i
        return None

    def position(self):
        """ This function return the position of the term in the dictionary index.
        
        --- Return ---
        position: int, the position of the term in the dictionary index"""
        return self._position
    
    def get_post(self, docid):
        """ This function return the posting of a document.
        
        --- Parameters ---
        docid: int, the id of the document to find in the posting list
        
        --- Return ---
        posting: tuple, the posting of the document (docid, tf)
        """
        index = self.find_posting_index(docid)    # find the index of the posting
        if index is not None:                     # if the posting is in the posting list
            return self._postings[index]          # return the posting
        else:                                     # if the posting is not in the posting list
            return None                           # return None
  
    


class Index():
    """ This class represent an index of a corpus"""
    def __init__(self, corpus):
        self._terms = OrderedDict()               # dictionary of terms
        self.make_index(corpus)                   # make the index
        self.number_of_documents = len(corpus)    # number of documents in the corpus
    
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
        post = self._terms[term].get_post(docid)      # get the posting of the term in the document
        if post is not None:                          # if the term is   in the document
            return post[1] 
        else:                                         # if the term is not in the document
            return 0
    
    def tf_idf(self, term: str, docid, N):
        """ This function compute the tf-idf of a term in a document"""

        return self.tf(term, docid)*self.idf(term)
        
    def make_index(self, corpus):
        """ This function make the index of the corpus"""
        position = 0                                            # position of the term in t_index
        for docid, doc in enumerate(tqdm(corpus, total=len(corpus), desc="Indexing:")):
            for term in doc:
                try:                                            # if the term is already in the index
                    self._terms[term].add_posting(docid)        #update the posting list
                except KeyError:                                # if the term is not in the index
                    self._terms[term] = Posting_list(position)
                    position += 1                               # update the position
                    self._terms[term].add_posting(docid)
    
    def remove_stopwords(self, percentage = 0.95):
        """ This function remove the terms that appear in more than percentage of the documents
        """
        for term in list(self._terms):                               # iterate over a copy of the keys
            if self.df(term) > percentage*self.number_of_documents:  # if the term appear in more than percentage of the documents
                del self._terms[term]                                # remove the term from the index
        self.update_positions()                                      # update the positions of the terms in the index

    def update_positions(self):
        """ This function update the positions of the terms in the index"""
        for i, (key, value) in enumerate(self._terms.items()):       # iterate over the terms
            value._position = i                                      # update the position of the term

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
    
    