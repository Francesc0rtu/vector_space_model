

class Posting():
    def __init__(self):
        self.doc_id = None
        self.tf = None

class Posting_list():
    def __init__(self, doc_id):
        self.posting_list = [doc_id]

    def append(self, doc_id):
        self.posting_list.append(doc_id)


class Term():
    ''' 
    '''

    def __init__(self, term, doc_id):
        self.term = term
        self.posting_list = Posting_list(doc_id)
        self.df = None
    
    def __str__(self):
        return f"term: {self.term}, df: {self.df}"

    def add_posting(self, doc_id):
        self.posting_list.append(doc_id)
    

class Index():

    def __init__(self):
        self.dict = {}  # * token -> Term(  ,   )
        self.doc_len = {}
