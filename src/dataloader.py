import re
from collections import defaultdict
from src.utils import preprocess_row
import ir_datasets


class Dataset():
    """This class is used to load the dataset and the query. """
    def __init__(self, dataset="time", path_query=None):
        

        if dataset == "time": 
            path='DATA/time/TIME.ALL'
            path_query='DATA/time/TIME.QUE'
            self.corpus = self.load_time(path)
            if path_query != None:
                self.query = self.load_time(path_query)
            else:
                self.query = "No query"
        
        if dataset == "cisi":
            path = 'DATA/cisi/CISI.ALL'
            path_query = 'DATA/cisi/CISI.QRY'
            self.corpus, self.query = self.load_cisi(path, path_query)
            self.rel = self.load_cisi_rel('DATA/cisi/CISI.REL')


            

    def __getitem__(self, key): 
        if key == 'corpus':
            return self.corpus
        elif key == 'query':
            return self.query
        elif key == 'rel':
            return self.rel
        else:
            raise KeyError
    
    


    def load_time(self, path):
        """Load the dataset time and return a list of articles. Each article is a list of words."""
        articles = [] # list of articles
        with open(path, 'r') as f:
            tmp = []
            for row in f: # for each row in the file
                if row.startswith("*FIND") or row.startswith("*TEXT"): # if the row is the beginning of a new article or new query
                    if tmp != []: # if the tmp is not empty
                        articles.append(tmp) # append the tmp to the articles
                    tmp = []
                else: # if the row is not the beginning of a new article or new query
                    tmp += preprocess_row(row)  # preprocess the row and append it to the tmp
        return articles
    
    def load_cisi(self, path, query_path):
        ### Processing DOCUMENTS
        doc_set = {}
        doc_id = ""
        doc_text = ""
        with open(path) as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
        doc_count = 0
        for l in lines:
            if l.startswith(".I"):
                doc_id = int(l.split(" ")[1].strip())-1
            elif l.startswith(".X"):
                doc_set[doc_id] = doc_text.lstrip(" ")
                doc_id = ""
                doc_text = ""
            else:
                doc_text += l.strip()[3:] + " " # The first 3 characters of a line can be ignored.    

        doc_set = [preprocess_row(doc) for doc in list(doc_set.values())]
        ### Processing QUERIES
        with open(query_path) as f:
            lines = ""
            for l in f.readlines():
                lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
            lines = lines.lstrip("\n").split("\n")
            
        qry_set = {}
        qry_id = ""
        for l in lines:
            if l.startswith(".I"):
                qry_id = int(l.split(" ")[1].strip()) -1
            elif l.startswith(".W"):
                qry_set[qry_id] = l.strip()[3:]
                qry_id = ""
        
        
        
        qry_set = [preprocess_row(qry) for qry in list(qry_set.values())]

        return doc_set, qry_set
    

    def load_cisi_rel(self, path):
        # ### Processing QRELS
        rel_set = {}
        with open(path) as f:
            for l in f.readlines():
                qry_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]) -1
                doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1])-1
                if qry_id in rel_set:
                    rel_set[qry_id].append(doc_id)
                else:
                    rel_set[qry_id] = []
                    rel_set[qry_id].append(doc_id)
        return rel_set