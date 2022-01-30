from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


def _list_to_dict(keys, values):
    return {key: value for key, value in zip(keys, values)}

def evaluate(list_of_references, hypotheses, keys=None):
    """Evaluate generated hypothesis sentences with (multiple) reference
    sentences.
    Assume N is the number of evaluated sentences in the corpus.
    list_of_references: list of length N, each of which is a list of references
    hypotheses: list of length N, each of which is a hypothesis
    Each reference or hypothesis sentence should be a str with space-separated
    tokens.
    This function wraps the pycocoevalcap package.
    See https://github.com/salaniz/pycocoevalcap for details of the package.
    Returns: score_dict, a dict with each key as the name of the metric and
    value the corresponding score. See the code for details of the metrics.
    """

    assert len(list_of_references) == len(hypotheses)

    # the scorers require the hypothesis also in a list
    hypotheses = [[hypothesis] for hypothesis in hypotheses]
    # the scorers require each argument in a dict
    if keys is None:
        keys = list(range(len(hypotheses)))
    list_of_references = _list_to_dict(keys, list_of_references)
    hypotheses = _list_to_dict(keys, hypotheses)

    # set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE"),
    ]

    # compute scores
    score_dict = {}
    def add_score(metric, score, scores):
        score_dict[metric] = score
    for scorer, metric in scorers:
        score, scores = scorer.compute_score(list_of_references, hypotheses)
        if isinstance(metric, list):
            for metric_, score_, scores_ in zip(metric, score, scores):
                add_score(metric_, score_, scores_)
        else:
            add_score(metric, score, scores)

    return score_dict
