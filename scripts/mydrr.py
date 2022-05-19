import json, os, sys
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer  # for nltk word tokenization
from nltk.stem.wordnet import WordNetLemmatizer
import fasttext.util
from collections import OrderedDict 

sort_flag = 1

## download bin file from https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
## and set path below 
config = json.load("config.json")
ft = fasttext.load_model(config["fasttext_model_loc"])
print("fastText load completed.", file = sys.stderr)
lmtzr = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r"(?x)(?:[A-Za-z]\.)+| \w+(?:\w+)*")
emb_size = 300

stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she','her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
    'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won',
    'wouldn'] + [str(i) for i in range(2100)] + ["one", "two", "three", "four", "five", 
    "six", "seven", "eight", "nine", "ten"] + ["00", "01", "02", "03", "04", "05", "06", 
    "07", "08", "09"] + ["once", "twice", "time"]
stop_words = [lmtzr.lemmatize(w1) for w1 in stop_words]
stop_words = list(set(stop_words))

# tokenization
def tokenization(sentences, stop_word_flag):
    words = tokenizer.tokenize(sentences.lower())
    words = [lmtzr.lemmatize(w1) for w1 in words]
    if stop_word_flag == 1:
        words = [w for w in words if w not in stop_words]
    return words

# term の並びから行列を生成する
def sent_Emb(ht_terms, embeddings_index, emb_size=300):
    HT_Matrix = np.empty((0, emb_size), float)
    tokens_not_found_embeddings = []
    tokens_embeddings_found = []
    for ht_term in ht_terms:
        coefs = np.asarray(
            embeddings_index.get_word_vector(ht_term),
            dtype='float32'
            )
        
        b = np.linalg.norm(coefs, ord=2) ## ||word_embedding(term)||_2
        if b != 0 :
            coefs = coefs / float(b)

        ## coefs = embeddings_index.get_word_vector(ht_term) / ||word_embedding(term)||_2

        HT_Matrix = np.append(HT_Matrix, np.array([coefs]), axis=0)
        tokens_embeddings_found.append(ht_term)
        
    return HT_Matrix, tokens_not_found_embeddings, tokens_embeddings_found

# score ベクトルの生成
def compute_alignment_vector(
        Hypo_matrix,
        hypo_toks_nf, ## always []
        hypo_toks_found,
        table_matrix,
        table_toks_nf, ## always []
        threshold=0.90):
    table_matrix = table_matrix.transpose()
    Score = np.matmul(Hypo_matrix, table_matrix)
    ## = [[h[0].t[0], h[0].t[1], ..., ]
    ##    [h[1].t[0], h[1].t[1], ..., ]
    ##    :]
    Score = np.sort(Score, axis=1)
    max_score1 = Score[:, -1:]  # taking the highest element column
    ## = [max(h[0].t[i]), max(h[1].t[i]), ..., ]^t
    max_score1 = np.asarray(max_score1).flatten()
    ## = [max(h[0].t[i]), max(h[1].t[i]), ..., ]
    ## 各 hypo_term に対し，table_term のうち最も高いベクトル内積値を採用

    max_score1 = [1 if s1 >= threshold else 0 for s1 in max_score1]
    remaining_terms = []

    for i1, s1 in enumerate(max_score1):
        if s1 == 0:
            remaining_terms.append(hypo_toks_found[i1])

    ## 各 hypo_term に対し，table_term のうち最も高いベクトル内積値が threshold = 0.90 以下のものを remaining_terms として返す
    return remaining_terms


def compute_alignment_score(Hypo_matrix, table_matrix):
    table_matrix = table_matrix.transpose()
    Score = np.matmul(Hypo_matrix, table_matrix)
    Score = np.sort(Score, axis=1)
    max_score1 = Score[:, -1:]  # taking highest element columns
    max_score1 = np.asarray(max_score1).flatten()

    ## 各 hypo_term に対し，table_term のうち最も高いベクトル内積値の和
    return np.sum(max_score1)


def get_alignment_justification(
        hypo_terms, ## terms in hypothesis sentence 
        table_k_v,  ## premises (List<String>)
        title,
        embedding_index, ## fasttext models
        threshold,  ## the number of adopted rows
        emb_size, ## 300
        flag=1):
    #print("hypo:", hypo_terms)

    Hypo_matrix, hypo_toks_nf, hypo_toks_found = sent_Emb(hypo_terms, embedding_index, emb_size) 
    ## hypo_toks_nf is always []
        
    ## Hypo_matrix = [
    ##   [ word_embedding(hypo_term[0]) / ||word_embedding(hypo_term[0])||_2 ] standarized
    ##   [ word_embedding(hypo_term[1]) / ||word_embedding(hypo_term[1])||_2 ]
    ##   :
    ## ]
    table_hyp_remaining_terms = {}
    Final_alignment_scores = []
    num_remaining_terms = []

    for ind, t_terms in enumerate(table_k_v):
        table_terms = tokenization(t_terms, 1)
        table_terms = list(set(table_terms) - set(title))
        #print("table {}".format(ind), table_terms)
        table_matrix, table_toks_nf, table_toks_found = sent_Emb(
            table_terms, embedding_index, emb_size) # table_toks_nf is always [] 
        ## table_matrix = [
        ##   [ word_embedding(table_term[0]) / ||word_embedding(table_term[0])||_2 ]
        ##   [ word_embedding(table_term[1]) / ||word_embedding(table_term[1])||_2 ]
        ##   :
        ## ]
        table_hyp_remaining_terms.update({ind: compute_alignment_vector(
            Hypo_matrix, hypo_toks_nf, hypo_toks_found, table_matrix, table_toks_nf)})  ## meaningless
        ind_score = compute_alignment_score(Hypo_matrix, table_matrix)
        num_remaining_terms.append(len(table_hyp_remaining_terms[ind]))  ## meaningless
        Final_alignment_scores.append(ind_score)

    # the higher alignment score, the more similar it is
    Final_index = list(np.argsort(Final_alignment_scores)[::-1])

    if flag == 0:  ## meaningless
        return Final_index[0], table_hyp_remaining_terms[Final_index[0]], table_hyp_remaining_terms
    else:
        if sort_flag=="0": final_list = {}
        else: final_list = OrderedDict()

        for i in range(min(len(Final_index), threshold)):
            final_list[Final_index[i]] = Final_alignment_scores[Final_index[i]]

        """
        final_list2 = {}
        for i in range(min([len(Final_index)])):
            final_list2[Final_index[i]] = Final_alignment_scores[Final_index[i]]
        #print(final_list2)
        # """

        ## final_list: Dict, table_k_v の各文で compute_alignment_score の大きいもののうち
        ## args["threshold"] = 4 個の table_k_v の index と score
        return final_list # , table_hyp_remaining_terms[Final_index[0]], table_hyp_remaining_terms


# Alignment over embeddings for sentence selection
def get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence(
        hypo_text, table_k_v, title, threshold, emb_size=300):
    embedding_index = ft
    hypo_terms = tokenization(hypo_text, 1)
    title_terms = tokenization(title, 1)
    hypo_terms = list(set(hypo_terms) - set(title_terms))

    if sort_flag == "0": indexes = set()
    else: indexes = []

    first_iteration_index1 = get_alignment_justification(
        hypo_terms, table_k_v, title_terms, embedding_index, threshold, emb_size, 1)

    for i in first_iteration_index1:
        if sort_flag == "0": indexes.add(i)
        else: indexes.append(i)

    return indexes, first_iteration_index1

# table_id, hypothesis -> List<
def adopted_premise(table_id, hypothesis, k):
    file_path = open(
        os.path.join("dataset/json/{}.json".format(table_id)),
        encoding='utf-8')

    json_data = json.load(file_path)
    title = json_data['title'][0]

    table_k_v = []
    keys = []

    para = ""
    for row in json_data:
        if row != "title":
            para += "{} {}\n".format(row, " ".join(json_data[row]))
            keys.append(row)
    
    sentences = para.split("\n")
    for sent in sentences:
        if sent != '': table_k_v.append(sent)

    ind, scores = get_iterative_alignment_justifications_non_parameteric_PARALLEL_evidence(hypothesis, table_k_v, title, k)

    ## scores is not used
    keys_adopted = []
    for i in ind:
        tokenized_key = " ".join(tokenization(keys[i], 1))
        tokenized_values = [
            " ".join(tokenization(value, 1)) for value in json_data[keys[i]]
        ]
        keys_adopted.append((tokenized_key, tokenized_values))

    return title, keys_adopted
