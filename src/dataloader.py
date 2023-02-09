import re
from collections import defaultdict
from src.utils import preprocess_row

class Dataset():
    def __init__(self, dataset="time", path_query=None):

        if dataset == "time":
            path='DATA/time/TIME.ALL'
            path_query='DATA/time/TIME.QUE'
            self.corpus = self.load(path)
            if path_query != None:
                self.query = self.load(path_query)
            else:
                self.query = "No query"

        #if dataset == "wikipedia":
            

    def __getitem__(self, key):
        if key == 'corpus':
            return self.corpus
        elif key == 'query':
            return self.query
        else:
            raise KeyError
        
    



    def load(self, path):
        articles = []
        with open(path, 'r') as f:
            tmp = []
            for row in f:
                if row.startswith("*FIND") or row.startswith("*TEXT"):
                    if tmp != []:
                        articles.append(tmp)
                    tmp = []
                else:
                    tmp += preprocess_row(row)
        return articles