import nltk_bleu_score
import sys

ref_file = sys.argv[1]
system_file = sys.argv[2]

systems = []
refs = []
with open(system_file, encoding='utf-8') as f:
    for line in f:
        if not line:
            break
        systems.append(line.strip().split(' '))

with open(ref_file, encoding='utf-8') as f:
    for line in f:
        if not line:
            break
        refs.append([line.strip().split(' ')])
bleu = nltk_bleu_score.corpus_bleu(refs, systems)
print('BLEU: {0}'.format(bleu))
