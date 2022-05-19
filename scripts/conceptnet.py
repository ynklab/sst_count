import requests
import json, re
import nltk_tools

# You can make 3600 requests per hour to the ConceptNet API, 
# with bursts of 120 requests per minute allowed. 
# The /related and /relatedness endpoints count as two requests when you call them.
# This means you should design your usage of the API to average less than 1 request per second.

dict = {}

def get_related_words(target, threshold):
    obj = requests.get('http://api.conceptnet.io/related/c/en/{}?filter=/c/en'.format(target)).json()
    rels = []
    for info in obj["related"]:
        id, weight = info["@id"], info["weight"]
        word = re.fullmatch(r'/c/en/(.+)', id).groups()[0]
        if weight > threshold: 
            word = nltk_tools.lemmatization(word)
            if word not in rels:
                rels.append(word)
    return rels

def relatedness(word1, word2):
    obj = requests.get('http://api.conceptnet.io/relatedness?node1=/c/en/{}&node2=/c/en/{}'.format(word1, word2)).json()
    return obj["value"]

def get_relatedness(word1, word2):
    if word1 > word2: word1, word2 = word2, word1
    
    if word1 in dict and word2 in dict[word1]:
        return dict[word1][word2]
    
    score = relatedness(word1, word2)
    if word1 not in dict: dict[word1] = {}
    if word2 not in dict[word1]: dict[word1][word2] = score
    return score

def export_score_list(filename):
    with open(filename, "w") as f:
        for word1 in dict:
            for word2 in dict[word1]:
                print(word1, word2, dict[word1][word2], sep = '\t', file = f)

def test():
    pass

if __name__ == "__main__":
    test()