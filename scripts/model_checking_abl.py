import os, sys, time
import nltk_tools
import spacy_tools
import conceptnet
import pandas as pd
from timeout_decorator import timeout, TimeoutError
import nltk.sem
#from nltk.sem import Valuation, Model, evaluate
from nltk.internals import Counter
import my_model_evaluate
#from my_model_evaluate import Valuation, Model
from misc import DictInDict
from nltk.sem.logic import Expression, LogicalExpressionException
import mynltk2normal 

TIME_LIMIT = 10
lexpr = Expression.fromstring
symmetric_verbs = ["marry"]

# starting from X0
class XCounter(Counter):
    def reset(self):
        self._value = 0
    def get(self): 
        ret = "X{}".format(self._value) 
        self._value += 1 
        return ret

# starting from V0
class VCounter(XCounter):
    def get(self): 
        ret = "V{}".format(self._value) 
        self._value += 1 
        return ret 


def construct_model(title, rows, hypo_sent, threshold, case_index, child_index):
    dict = DictInDict()
    dict_sub = DictInDict()
    filepath = "experiments/logs/relatedness/{:>03}/{:>03}_{:>02}.tsv".format(case_index, case_index, child_index)
    #dict.import_from_tsv(filepath)
    dict_sub.import_from_tsv("experiments/logs/relatedness/all.tsv")

    x_counter = XCounter()
    v_counter = VCounter()

    title = title.lower().split(",")[0].replace(" ", "")
    hypo_terms = nltk_tools.tokenization(hypo_sent, 2)
    all_variables = []

    title_entity = x_counter.get()
    valuation = [(title, set([title_entity]))]
    if title[:3] == "the": 
        valuation.append((title[3:], set([title_entity])))
    else:
        valuation.append(("the" + title, set([title_entity])))
    all_variables.append(title_entity)

    # binary predicate Subj(e, x): subjective
    sbjs = []
    # binary predicate Acc(e, x): accusative
    accs = []

    # shared event "have"
    have_event = v_counter.get()
    # META Unary predicate have(e)
    have_events = [have_event]

    for row in rows:
        start = time.time()
        key_phrase = row[0]
        key_terms = nltk_tools.tokenization(key_phrase, 2)
        values = [v.replace(" ", "") for v in row[1]]
        
        key_tag = spacy_tools.get_tag(key_phrase)
        entities = []
        events = []

        for value in values:
            entity = x_counter.get()
            entities.append(entity)
            event = v_counter.get()
            events.append(event)
            valuation.append((value, set([entity])))



        if key_tag == "VBN": # Verb, past participle
            for event in events:
                for entity in entities:
                    sbjs.append((event, entity))
                    accs.append((event, title_entity))
                    if key_terms[0] in symmetric_verbs:
                        sbjs.append((event, title_entity))
                        accs.append((event, entity))
            for key_term in key_terms:
                valuation.append((key_term, set(events)))

        elif key_tag[0] == 'V': # Verb except VBN
            for event in events:
                for entity in entities:
                    sbjs.append((event, title_entity))
                    accs.append((event, entity))
                    if key_terms[0] in symmetric_verbs:
                        sbjs.append((event, entity))
                        accs.append((event, title_entity))
            for key_term in key_terms:
                valuation.append((key_term, set(events)))

        else: # key_tag[0] == 'N': # Noun
            for entity in entities:
                sbjs.append((have_event, title_entity))
                accs.append((have_event, entity))
            for key_term in key_terms:
                valuation.append((key_term, set(entities)))
        

        # virtual axioms
        for key_term in key_terms:
            for hypo_term in hypo_terms:
                if dict_sub.exist(key_term, hypo_term):
                    score = dict_sub.get(key_term, hypo_term)
                    dict.insert(key_term, hypo_term, score)
                else:
                    score = conceptnet.get_relatedness(key_term, hypo_term)
                    dict.insert(key_term, hypo_term, score)

                if score > threshold:
                    lem, pos = spacy_tools.get_pos(hypo_term)
                    if pos == "NOUN":
                        valuation.append((lem, set(entities)))
                    elif pos == "VERB":
                        valuation.append((lem, set(events)))
                        if key_tag == "VBN":
                            for event in events:
                                accs.append((event, title_entity))
                                for entity in entities:
                                    sbjs.append((event, entity))
                                if lem in symmetric_verbs:
                                    sbjs.append((event, title_entity))
                                    for entity in entities:
                                        accs.append((event, entity))
                                    
                        else:
                            for event in events:
                                sbjs.append((event, title_entity))
                                for entity in entities:
                                    accs.append((event, entity))
                                if lem in symmetric_verbs:
                                    accs.append((event, title_entity))
                                    for entity in entities:
                                        sbjs.append((event, entity))


    valuation.append(("have", set(have_events)))
    valuation.append(("Subj", set(sbjs)))
    valuation.append(("Acc", set(accs)))

    # base
    val_ = nltk.sem.Valuation(valuation)
    dom = val_.domain

    val = nltk.sem.Valuation(valuation + [("True", dom)])
    m_base = nltk.sem.Model(dom, val)
    
    # opt
    val_ = my_model_evaluate.Valuation(valuation)
    dom = val_.domain

    val = my_model_evaluate.Valuation(valuation + [("True", dom)])
    m_opt = my_model_evaluate.Model(dom, val)

    # entity: X
    # event: V
    dict.export_to_tsv(filepath)

    return (m_base, m_opt)

def formula_rev(formula):
    mynltk2normal.counter_reset()
    return str(mynltk2normal.rename_variable(mynltk2normal.remove_true(lexpr(formula))))

@timeout(TIME_LIMIT)
def model_evaluate(model, formula, g):
    start = time.time()
    ret = model.evaluate(formula, g)
    return ret, time.time() - start

def model_evaluate_inf_opt(model, formula, g):
    start = time.time()
    ret = model.evaluate(formula, g)
    #print("opt", time.time() - start, ret)
    return ret, time.time() - start
    
def model_evaluate_inf_base(model, formula, g):
    start = time.time()
    ret = model.evaluate(formula, g)
    #print("base", time.time() - start, ret)
    return ret, time.time() - start


def model_check_abl(model_base, model_opt, formulas):
    t_base = 0
    t_opt = 0

    formula = formula_rev(formulas[0])

    g = my_model_evaluate.Assignment(model_opt.domain)
    res, t = model_evaluate_inf_opt(model_opt, formula, g)
    t_opt += t
    
    g = nltk.sem.Assignment(model_base.domain)
    res, t = model_evaluate_inf_base(model_base, formula, g)
    t_base += t

    return (t_opt, t_base)

    
def main():
    pass


if __name__ == "__main__":
    main()