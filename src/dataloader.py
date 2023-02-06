import re
from collections import defaultdict


class Dataset():
    def __init__(self, path='DATA/TIME.ALL', path_query='DATA/TIME.QUE'):
        self.corpus = self.load(path)
        if path_query != None:
            self.query = self.load(path_query)

    def __call__(self):
        return self.corpus
    
    def load(self, path):
        articles = []
        with open(path, 'r') as f:
            tmp = []
            for row in f:
                if row.startswith("*TEXT"):
                    if tmp != []:
                        articles.append(tmp)
                    tmp = []
                else:
                    row = re.sub(r'[^a-zA-Z\s]+', '', row)
                    tmp += row.split()
        return articles