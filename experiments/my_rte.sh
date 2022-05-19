#!/bin/bash

function timeout() { 
  perl -e 'alarm shift; exec @ARGV' "$@"; 
}

function remove_empty_file() {
  c=`wc -m $1 | awk '{print $1'}`
  if [ $c = 0 ]; then
    rm $1
  fi
}

# Usage
# ./my_rte.sh "There are three books." semantic_template_event.yaml 3 000_00"
USAGE="Usage: ./my_rte.sh \"sentence\" <semantic_templates.yaml> <nbest> <filepath>"

# Set the number of nbest parses
nbest=$3

configJson=`cat config.json`
c2l_dir=$(echo $configJson | jq -r '.ccg2lambda_loc')
candc_dir=$(echo $configJson | jq -r '.candc_loc')
tregex_dir=$(echo $configJson | jq -r '.tregex_loc')

# Set tregex path:
# tregex_dir="/home/ast/.local/stanford-tregex-2018-10-16"
# export CLASSPATH=$tregex_dir/stanford-tregex.jar:$CLASSPATH

# This variable contains the filename where the semantic templates are.
semantic_templates=$2
if [ ! -f $semantic_templates ]; then
  echo "Error: File with semantic templates does not exist."
  echo $USAGE
  exit 1
fi

sentence=$1

# These variables contain the names of the directories where intermediate
# results will be written.
# sentence_basename=$(echo ${sentence} | sed 's/ /_/g')
# template_suffix=$(echo ${semantic_templates##*_})
# template_suffix=$(echo ${template_suffix%%.*})
# sentence_basename=${sentence_basename}${template_suffix}
sentence_basename=$4

plain_dir="experiments/logs/cache"       # tokenized sentences.
parsed_dir="experiments/logs/cache"      # parsed sentences into XML or other formats.
results_dir="experiments/logs/semantics" # parsed result
error_dir="experiments/logs/error"

#mkdir -p $plain_dir $parsed_dir $results_dir $error_dir

# Tokenize text with Penn Treebank tokenizer.
echo $sentence | \
  sed -f ${c2l_dir}/en/tokenizer.sed | \
  sed 's/ _ /_/g' | \
  sed 's/[[:space:]]*$//' | \
  sed 's/ (//g' | \
  sed 's/) //g' \
  > ${plain_dir}/${sentence_basename}.tok

depccg_exists=`pip freeze | grep depccg`
if [ "${depccg_exists}" == "" ]; then
    echo "depccg parser directory incorrect. Exit."
    exit 1
fi

function parse_candc() {
  # Parse using C&C.
  ${candc_dir}/bin/candc \
      --models ${candc_dir}/models \
      --candc-printer xml \
      --input ${plain_dir}/${sentence_basename}.tok \
    2> ${parsed_dir}/${sentence_basename}.log \
     > ${parsed_dir}/${sentence_basename}.candc.xml
  python ${c2l_dir}/en/candc2transccg.py ${parsed_dir}/${sentence_basename}.candc.xml \
    > ${parsed_dir}/${sentence_basename}.jigg.xml \
    2>> ${parsed_dir}/${sentence_basename}.log
}

function parse_depccg() {
  cat ${plain_dir}/${sentence_basename}.tok | \
  env CANDC=${candc_dir} depccg_en \
    --input-format raw \
    --annotator spacy \
    --nbest $nbest \
    --format jigg_xml \
  > ${parsed_dir}/${sentence_basename}.init.jigg.xml \
  2> ${parsed_dir}/${sentence_basename}.log
}

function tsurgeon() {

  python scripts/brackets_with_pos.py ${parsed_dir}/${sentence_basename}.init.jigg.xml \
    > ${parsed_dir}/${sentence_basename}.ptb
  export CLASSPATH=$tregex_dir/stanford-tregex.jar:$CLASSPATH
  java -mx100m edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon -s \
    -treeFile ${parsed_dir}/${sentence_basename}.ptb experiments/transform.tsgn \
    > ${parsed_dir}/${sentence_basename}.tsgn.ptb
}

function semantic_parsing() {
  sentence=$(cat ${plain_dir}/${sentence_basename}.tok)

  python ${c2l_dir}/scripts/semparse.py \
    $parsed_dir/${sentence_basename}.jigg.xml \
    $semantic_templates \
    $results_dir/${sentence_basename}.sem.xml \
    2> $error_dir/${sentence_basename}.sem.err

  remove_empty_file $error_dir/${sentence_basename}.sem.err
  python ${c2l_dir}/scripts/visualize.py $results_dir/${sentence_basename}.sem.xml \
    > $results_dir/${sentence_basename}.html

}

# CCG parsing, semantic parsing and theorem proving
# parser_name="depccg"
parse_depccg 
# parse_candc

#python change_tags.py ${parsed_dir}/${sentence_basename}.init.jigg.xml
tsurgeon 

cat ${parsed_dir}/${sentence_basename}.tsgn.ptb \
  | sed -e 's/\((\|\\\|\/\|<\|>\)\(S\|N\|NP\|PP\)\([a-zX,=]\+\)/\1\2[\3]/g' \
  | sed -e 's/\(\/\)\(S\|N\|NP\|PP\)\(\[\)\([a-zX,=]\+\)\(\]\)\()\)/\1\2\4\6/g' \
  | sed 's/</(/g' | sed 's/>/)/g' \
  | sed 's/[:\|;];/\.;/g' \
  | sed -e 's/;[.,\$_A-Z-]\+\$*//g' \
  > ${parsed_dir}/${sentence_basename}.tsgn.mod.ptb
python scripts/tagger.py --format jigg_xml ${parsed_dir}/${sentence_basename}.tsgn.mod.ptb \
  > ${parsed_dir}/${sentence_basename}.jigg.xml \
  2> ${parsed_dir}/${sentence_basename}.jigg.xml.tagger.err.txt
#python change_tags.py ${parsed_dir}/${sentence_basename}.jigg.xml

#echo "semantic parsing" >&2
semantic_parsing

python scripts/extract_formula_from_xml.py ${results_dir}/${sentence_basename}.sem.xml #\
#  > ${results_dir}/${sentence_basename}.${parser_name}.out

