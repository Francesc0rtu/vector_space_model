import os
import pandas as pd
import subprocess

class dataset():
    """This class is used to load the dataset from the csv file. """
    def __init__(self, dataset_path = "data/wikIR59k/documents.csv"):
        if not os.path.exists(dataset_path):
            download_from_zenodo()
        self.dataset = pd.read_csv(dataset_path)
  
    def get_dataset(self):
        return self.dataset
    def get_documents(self):
        return self.dataset['document']
    def get_document(self, doc_id):
        return self.dataset['document'][doc_id]



def download_from_zenodo(doi = "10.5281/zenodo.3557342", path = "data/"):
    """Downloads the dataset from Zenodo and unzips it into the specified path.
    for now it only works for the wikIR59k dataset, but it can be easily extended to other datasets."""


    destination_path = os.path.join(os.getcwd(), path)

    print(f"Downloading dataset from Zenodo into {destination_path}... This could take a while...")

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    os.chdir( destination_path )
    os.system( f"zenodo_get {doi}" )

    print("Unzipping...")

    os.system( f"unzip *.zip" )
    os.system( f"rm *.zip" )

    os.chdir("../")
