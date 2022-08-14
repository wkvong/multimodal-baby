import itertools 
import spacy


nlp = None


def tokenize(s : str, kind='space'):
    if kind == 'spacy':
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")
        return nlp(s)
    elif kind == 'space':
        return s.split()
    else:
        raise Exception("Unrecognized kind='{kind}'")


def untokenize(s):
    if isinstance(s, spacy.tokens.doc.Doc):
        return str(s)
    else:
        return ' '.join(s)


def paired(objects, n=2):
    objects = iter(objects)
    while True:
        try:
            yield tuple([next(objects) for i in range(n)])
        except StopIteration:
            break


def unpaired(paired_objects):
    return itertools.chain.from_iterable(paired_objects)


def read_sentences_and_losses(file):
    with open(file, 'r') as f:
        for line in f:
            sentence, _, loss = line.rstrip().rpartition(' ')
            yield sentence.rstrip(), float(loss)