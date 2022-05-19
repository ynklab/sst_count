import os
import pandas as pd

class DictInDict(dict):
    def __init__(self):
        self.dict = {}

    def insert(self, key1, key2, value):
        if key1 > key2: key1, key2 = key2, key1
        if key1 not in self.dict: self.dict[key1] = {}
        if key2 not in self.dict[key1]: self.dict[key1][key2] = value
    
    def exist(self, key1, key2):
        if key1 > key2: key1, key2 = key2, key1
        if key1 not in self.dict: return False
        if key2 not in self.dict[key1]: return False
        return True

    def get(self, key1, key2):
        if key1 > key2: key1, key2 = key2, key1
        if key1 in self.dict and key2 in self.dict[key1]: return self.dict[key1][key2]
        raise KeyError

    def import_from_df(self, df):
        for i in range(len(df)):
            self.insert(df["word1"][i], df["word2"][i], df["score"][i])

    def import_from_tsv(self, filepath):
        if filepath != None and os.path.exists(filepath):
            df = pd.read_csv(filepath, sep = '\t')
            self.import_from_df(df)
    
    def export_as_df(self):
        arr = []
        for word1 in self.dict:
            for word2 in self.dict[word1]:
                arr.append([word1, word2, self.dict[word1][word2]])
        return pd.DataFrame(arr, columns=['word1', 'word2', 'score'])
    
    def export_to_tsv(self, filepath):
        d = list(map(str, filepath.split("/")))
        filedir = "/".join(d[:-1])
        os.makedirs(filedir, exist_ok=True)
        df = self.export_as_df()
        df.to_csv(filepath, index=False, sep='\t')