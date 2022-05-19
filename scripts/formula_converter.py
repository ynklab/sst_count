import argparse, json, os, re, sys
import subprocess
import spacy_tools
from nltk.sem.logic import Expression


def semantic_parsing(sentence, yaml_suffix, nbest, case_index, child_index):
    sentence = spacy_tools.convert_sentence_with_ner(sentence)
    sem_temp = "experiments/semantic_template.yaml"
    os.makedirs("experiments/logs/semantics/{:>03}".format(case_index), exist_ok=True)
    os.makedirs("experiments/logs/cache/{:>03}".format(case_index), exist_ok=True)
    os.makedirs("experiments/logs/error/{:>03}".format(case_index), exist_ok=True)
    filepath = "{0:>03}/{0:>03}_{1:>02}".format(case_index, child_index)
    
    o = subprocess.check_output((
        './experiments/my_rte.sh', 
        "{}".format(sentence), 
        sem_temp, 
        str(nbest), 
        filepath))
        
    return list(map(str, o.decode().strip().split('\n')))

def export_tree_as_html(case_index, child_index):
    sem_filepath = "experiments/logs/semantics/{:>03}/{:>03}_{:>02}.sem.xml".format(case_index, case_index, child_index)
    html_filepath = "experiments/logs/semantics/{:>03}/{:>03}_{:>02}.html".format(case_index, case_index, child_index)
    config = json.load("config.json")
    o = subprocess.check_output((
        'python', 
        "{}/scripts/visualize.py".format(config["ccg2lambda_loc"]), 
        sem_filepath))

    with open(html_filepath, "w") as f:
        print(o.decode().strip(), file = f)

def test():
    return None
    
if __name__ == "__main__":
    export_tree_as_html(sys.argv[1], sys.argv[2])