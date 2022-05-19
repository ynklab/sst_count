import argparse, datetime, os, sys, time
import pandas as pd
from nltk.sem.logic import Expression
sys.path.append("./scripts")
import mydrr, model_checking, formula_converter

# nltk 3.1 

lexpr = Expression.fromstring
cols = ["index", "case_index", "child_index", "table_id", "hypothesis", "gold", "time", "result"]

##########################
### Experiment Setting ###
parse = 0
rows = 2
threshold = 0.5
nbest = 1
# time_limit = 10
##########################
##########################

def main(args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./dataset/testcases.tsv")
    #parser.add_argument("--config", type=str, default="../../config/")
    #parser.add_argument("--log", type=str, default="logs/")
    parser.add_argument("--output", type=str, default="output_gen.tsv")      
    args = parser.parse_args()


    df = pd.read_csv(args.dataset, sep = "\t")
    rets = []

    with open(args.output, "w") as f0:
        for i in range(len(df)):
            start = time.time()

            case_index, child_index = map(int, df["case_index"][i].split("-"))
            table_id = df["table_id"][i]
            hypothesis = df["hypothesis"][i]
            label = df["label"][i]

            if i == 0:
                print(*cols, sep = "\t", file = f0)

            dt_now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            os.makedirs("experiments/logs/main_log", exist_ok=True)
            os.makedirs("experiments/logs/main_log/{:>03}".format(case_index), exist_ok=True)
            filepath = "experiments/logs/main_log/{0:>03}/{0:>03}_{1:>02}.txt".format(case_index, child_index)

            print(i, case_index, child_index, dt_now, label, sep = '\t', end = '\t', file = sys.stderr)

            with open(filepath, "w") as f:
                print(dt_now, file = f)

                if parse == 0:
                    # extract some rows to judge truth-values
                    title, adopted_premise = mydrr.adopted_premise(table_id, hypothesis, rows)
                    print(*adopted_premise, sep = '\n', file = f)

                    # construct a model from rows
                    premise_fol = model_checking.construct_model(title, adopted_premise, hypothesis, threshold, case_index, child_index)
                    print(premise_fol, file = f)

                # get semantics of hypothesis sentence by ccg2lambda
                conclusion_formulas = formula_converter.semantic_parsing(hypothesis, "orig0", nbest, case_index, child_index)
                print(*conclusion_formulas, sep = '\n', file = f)

                if parse == 0:
                    ret, t = model_checking.model_check(premise_fol, conclusion_formulas)
                    end = time.time()
                    while len(ret) < nbest: ret.append('None')
                    print(*ret, file = sys.stderr)
                    print(i, case_index, child_index, table_id, hypothesis, label, round(end-start, 3), *ret, sep = '\t', file = f)
                    print(i, case_index, child_index, table_id, hypothesis, label, round(end-start, 3), *ret, sep = '\t', file = f0)

            #break

if __name__ == '__main__':
    main()
