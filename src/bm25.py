from text_processing import *

class Posting():
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.tf = 1
    
    def update_tf(self ):
        self.tf = self.tf + 1

class Posting_list():
    def __init__(self, doc_id):
        self.posting_list = [Posting(doc_id)]

    def append(self, doc_id):
        self.posting_list.append(Posting(doc_id))
    
    def get_Posting(self, doc_id):
        for post in self.posting_list:
            if post.doc_id == doc_id:
                return post
        
        return None



class Term():

    def __init__(self, term, doc_id):
        self.term = term
        self.posting_list = Posting_list(doc_id)
        self.df = None
    
    def __str__(self):
        return f"term: {self.term}, df: {self.df}"
    
    def check_Posting(self, doc_id):
        posting  = self.posting_list.get_Posting(doc_id)
        if posting is None:
            self.add_Posting(doc_id)
        else:
            posting.update_tf()

    def add_Posting(self, doc_id):
        self.posting_list.append(doc_id)
    

class Index():

    def __init__(self):
        self.dict = {}  # * token -> Term(  ,   )
        self.doc_len = {}

    def Create_Index(self, corpus):
        

