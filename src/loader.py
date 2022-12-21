import os
import csv
import subprocess
from functools import total_ordering, reduce
import re

class Corpus():
    """This class is used to load the dataset from the csv file. """
    def __init__(self, dataset_path = "data/wikIR59k/documents.csv"):
        
        print( "Loading dataset... if you don't have it, it will be downloaded from Zenodo. This could take up to 15 minutes depending on your internet connection.")

        if not os.path.exists(dataset_path):
            download_from_zenodo(path = "data/")
        self.corpus = []
        with open(dataset_path, 'r', encoding = "utf8") as csv_file:
            documents = csv.reader(csv_file, delimiter = ",")
            for doc in documents:
                self.corpus.append(doc[1])
        self.corpus.pop(0)
        print("Dataset loaded!")

    def get_doc(self, index):
        return self.corpus[index]
    def get_corpus(self):
        return self.corpus
    
    def tokenize_doc(self, doc_id):
        return self.corpus[doc_id].split() 

@total_ordering
class Posting:
    
    def __init__(self, doc_id):
        self.doc_id = doc_id
        
    def get_from_corpus(self, corpus):
        return corpus.get_doc(self.doc_id)
    
    def __eq__(self, other):
        return self.doc_id == other.doc_id
    
    def __gt__(self, other):
        return self.doc_id > other.doc_id

class Posting_list:

    def __init__(self):
        self.postings = []
    
    def add_posting(self, posting):
        self.postings.append(posting)
    
    def get_postings(self):
        return self.postings
    
    def get_posting(self, index):
        return self.postings[index]


class Term:

    def __init__(self, term, doc_id):
        self.term = term
        self.posting_list = Posting_list()
    
    def add_posting(self, posting):
        self.posting_list.add_posting(posting)
           
def normalize(text):
    no_punctuation = re.sub(r'[^\w^\s^-]','',text)
    downcase = no_punctuation.lower()
    return downcase





def download_from_zenodo(doi = "10.5281/zenodo.3557342", path = None):
    """Downloads the dataset from Zenodo and unzips it into the specified path.
    for now it only works for the wikIR59k dataset, but it can be easily extended to other datasets."""

    if path is None:
        path = os.path.join( os.getcwd(), "data/") 
    

    print(f"Downloading dataset from Zenodo into {path}... This could take a while...")

    if not os.path.exists(path):
        os.makedirs(path)
    os.chdir( path )
    os.system( f"zenodo_get {doi}" )

    print("Unzipping...")

    os.system( f"unzip *.zip" )
    os.system( f"rm *.zip" )

    os.chdir("../")
