import itertools
import spacy


def tokenize(s, kind='spacy'):
    if kind == 'spacy':
        nlp = spacy.load(
            'en_core_web_sm',
            exclude=[
                'attribute_ruler', 'lemmatizer', 'ner',
                'senter', 'parser', 'tagger', 'tok2vec']
        )
        tokenizer = nlp.tokenizer
        if isinstance(s, str):
            return tokenizer(s)
        else:
            return tokenizer.pipe(s)
    elif kind == 'space':
        if isinstance(s, str):
            return s.split()
        else:
            return (sent.split() for sent in s)
    else:
        raise Exception(f"Unrecognized {kind=}")


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
