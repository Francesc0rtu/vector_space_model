import re
from collections import defaultdict


class Dataset():
    def __init__(self, path='DATA/TIME.ALL', path_query='DATA/TIME.QUE'):
        self.corpus = self.load(path)
        if path_query != None:
            self.query = self.load(path_query)

    def make_inverted_index(self):
        """
        This function builds an inverted index as an hash table (dictionary)
        where the keys are the terms and the values are ordered lists of
        docIDs containing the term.
        """
        index = defaultdict(set)
        for docid, article in enumerate(self.corpus):
            for term in article:
                index[term].add(docid)
        return index
    
    def make_positional_index(self):
        """
        A more advanced version of make_inverted_index. Here each posting is
        non only a document id, but a list of positions where the term is
        contained in the article.
        """
        index = defaultdict(dict)
        for docid, article in enumerate(self.corpus):
            for pos, term in enumerate(article):
                try:
                    index[term][docid].append(pos)
                except KeyError:
                    index[term][docid] = [pos]
        return index
    
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