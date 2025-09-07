#!/bin/bash

# automatic annotations 4o
python src/annotations/annotate_gpt_4o.py --num-txt-var 1 -m 'text' -b 'src/annotations/jsonl/annot_openai_4o' && \
python src/annotations/annotate_gpt_4o.py --num-txt-var 3 -m 'text' -b 'src/annotations/jsonl/annot_openai_4o' && \
python src/annotations/annotate_gpt_4o.py --num-txt-var 4 -m 'text' -b 'src/annotations/jsonl/annot_openai_4o' && \
python src/annotations/annotate_gpt_4o.py -m 'ts' -b 'src/annotations/jsonl/annot_openai_4o' && \
python src/annotations/annotate_gpt_4o.py -m 'ts3' -b 'src/annotations/jsonl/annot_openai_4o' && \
python src/annotations/annotate_gpt_4o.py -m 'ts4' -b 'src/annotations/jsonl/annot_openai_4o' && \

sh src/annotations/filter_annot_overlap.sh src/annotations/jsonl/

