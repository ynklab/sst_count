import argparse, datetime, json, os, re, sys, time
import numpy as np
import pandas as pd
from nltk.sem.logic import Expression
sys.path.append("./scripts")
import mydrr, model_checking_abl, formula_converter

# nltk 3.1 

lexpr = Expression.fromstring

##########################
### Experiment Setting ###
parse = 0
rows = 2
threshold = 0.5
nbest = 1
##########################
##########################


def main(args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./dataset/testcases.tsv")
    #parser.add_argument("--config", type=str, default="../../config/")
    #parser.add_argument("--log", type=str, default="logs/")
    parser.add_argument("--output", type=str, default="output_time.tsv")      
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, sep = "\t")
    rets = []

    for prob_set in [0, 1, 30, 84, 23, 24]:
        t_opt_l, t_base_l = [], []
        cnt = 0
        for i in range(len(df)):
            start = time.time()

            case_index, child_index = map(int, df["case_index"][i].split("-"))
            if case_index != prob_set: continue
            table_id = df["table_id"][i]
            hypothesis = df["hypothesis"][i]
            label = df["label"][i]


            dt_now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

            # extract some rows to judge truth-values
            title, adopted_premise = mydrr.adopted_premise(table_id, hypothesis, rows)
            if child_index == 0: print(adopted_premise, file = sys.stderr)

            # get semantics of hypothesis sentence by ccg2lambda
            conclusion_formulas = formula_converter.semantic_parsing(hypothesis, "orig0", nbest, case_index, child_index)

            t_opt_sum = 0
            t_base_sum = 0
            for _ in range(1):
                # construct a model from rows
                model_base, model_opt = model_checking_abl.construct_model(title, adopted_premise, hypothesis, threshold, case_index, child_index)
                if child_index == 0: print(model_base.domain, model_opt.domain, file = sys.stderr, sep = '\t')

                t_opt, t_base = model_checking_abl.model_check_abl(model_base, model_opt, conclusion_formulas)
                t_opt_sum += t_opt
                t_base_sum += t_base
            t_opt_l.append(t_opt_sum)
            t_base_l.append(t_base_sum)
            #print(prob_set, child_index, "opt", t_opt_sum, sep = '\t', file = sys.stderr)
            #print(prob_set, child_index, "base", t_base_sum, sep = '\t', file = sys.stderr)
            print(prob_set, child_index, 1, t_opt_sum, sep = '\t')
            print(prob_set, child_index, 0, t_base_sum, sep = '\t')
            cnt += 1

        #print(prob_set, "opt", sum(t_opt_l) / cnt, sep = '\t')
        #print(prob_set, "opt", sum(t_opt_l) / cnt, t_opt_l, sep = '\t', file = sys.stderr)
        #print(prob_set, "base", sum(t_base_l) / cnt, sep = '\t')
        #print(prob_set, "base", sum(t_base_l) / cnt, t_base_l, sep = '\t', file = sys.stderr)

if __name__ == '__main__':
    main()
