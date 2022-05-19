import re, sys
import spacy

nlp = spacy.load("en_core_web_sm")

def convert_sentence_with_ner(sentence):
    doc = nlp(sentence)
    offset = 0

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "LOC", "GPE", "NORP"]:
            s, e = ent.start_char + offset, ent.end_char + offset
            sentence = sentence[:s] + sentence[s:e].replace(" ", "_") + "_" + sentence[e:]
            offset += 1
    """
    comparatives = [
        "no less than",
        "no more than",
        "less than",
        "more than",
        "at least",
    ]

    for comp in comparatives:
        if comp in sentence:
            sentence = sentence.replace(comp, comp.replace(" ", "~"))
    """

    return sentence

def get_pos(word):
    doc = nlp(word)
    return doc[0].lemma_, doc[0].pos_

def get_tag(phrase):
    doc = nlp(phrase)
    return doc[0].tag_

def pos_tagging(word):
    doc = nlp(word)

    # 12.	NN	Noun, singular or mass [chip]
    # 13.	NNS	Noun, plural [chips]
    # 14.	NNP	Proper noun, singular [Microsoft]
    # 15.	NNPS	Proper noun, plural 
    # 27.	VB	Verb, base form [speak]
    # 28.	VBD	Verb, past tense [spoke]
    # 29.	VBG	Verb, gerund or present participle [speaking]
    # 30.	VBN	Verb, past participle [spoken]
    # 31.	VBP	Verb, non-3rd person singular present [speak]
    # 32.	VBZ	Verb, 3rd person singular present [speaks]

    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)

if __name__ == "__main__":
    pos_tagging("Conjectured in	1985")
