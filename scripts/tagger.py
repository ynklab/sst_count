import argparse, sys
import xml.etree.ElementTree as ET
from depccg.tools.reader import read_trees_guess_extension
from depccg.printer import ConvertToJiggXML
from depccg.tokens import english_annotator
from depccg.download import SEMANTIC_TEMPLATES
from lxml import etree

LANG = 'en'

def get_init_tags(filename):

    tree = ET.parse(filename)
    root = tree.getroot()

    lst = []
    for token in root.iter('token'):
        dic = {'word': token.attrib['surf'], 'pos': token.attrib['pos'],
               'entity': token.attrib['entity'],
               'lemma': token.attrib['base']}
        lst.append(dic)
    return lst


def to_jigg_xml(trees, tagged, use_symbol=False):
    root_node = etree.Element('root')
    document_node = etree.SubElement(root_node, 'document')
    sentences_node = etree.SubElement(document_node, 'sentences')
    sentence_node = etree.SubElement(sentences_node, 'sentence')
    tokens_node = etree.SubElement(sentence_node, 'tokens')

    cats = [leaf.cat for leaf in trees[0][0][0].leaves]
    assert len(cats) == len(tagged)

    for j, (token, cat) in enumerate(zip(tagged, cats)):
        token_node = etree.SubElement(tokens_node, 'token')
        token_node.set('start', str(j))
        token_node.set('cat', str(cat))
        token_node.set('id', f's{0}_{j}')
        if 'word' in token:
            token['surf'] = token.pop('word')
        if 'lemma' in token:
            token['base'] = token.pop('lemma')
        for k, v in token.items():
            token_node.set(k, v)
    
    converter = ConvertToJiggXML(0, use_symbol)
    for j in range(len(trees)):
        for tree, _ in trees[j]:
            sentence_node.append(converter.process(tree))

    #for i, parsed in enumerate(trees):
        
    return root_node

def print_(nbest_trees,
           tagged_doc,
           lang = 'en',
           format = 'auto',
           semantic_templates = None,
           file = sys.stdout):

    def process_xml(xml_node):
        return etree.tostring(xml_node, encoding='utf-8', pretty_print=True).decode('utf-8')

    use_symbol = lang == 'ja'
    print(process_xml(to_jigg_xml(nbest_trees, tagged_doc, use_symbol=use_symbol)), file=file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH',
                        help='path to either of *.auto, *.xml, *.jigg.xml, *.ptb')
    parser.add_argument('--annotator',
                        default='spacy',
                        choices=english_annotator.keys(),
                        help='annotate POS, named entity, and lemmas using this library')
    parser.add_argument('-f',
                        '--format',
                        default='xml',
                        choices=['auto', 'xml', 'prolog', 'jigg_xml', 'jigg_xml_ccg2lambda', 'json'],
                        help='input parser type')
    parser.add_argument('--semantic-templates',
                        help='semantic templates used in "ccg2lambda" format output')
    args = parser.parse_args()
    
    doc, trees = [], []
    for _, tokens, tree in read_trees_guess_extension(args.PATH):
        doc.append([token.word for token in tokens])
        trees.append([(tree, 0)])

    filename = args.PATH.replace('tsgn.mod.ptb', 'init.jigg.xml')
    tagged_doc = get_init_tags(filename)

    semantic_templates = args.semantic_templates or SEMANTIC_TEMPLATES[LANG]
    print_(trees,
           tagged_doc,
           format=args.format,
           lang=LANG)

