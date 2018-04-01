import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import warnings
from collections import defaultdict


def measure_quality(references_file_name, sources_file_name, hypotheses_file_name):
    with open(references_file_name) as references_handler,\
         open(sources_file_name) as sources_handler,\
         open(hypotheses_file_name) as hypotheses_handler:
                references_dict = defaultdict(list)
                hypotheses_dict = {}
                for reference, source, hypothesis in zip(references_handler, sources_handler, hypotheses_handler):
                    references_dict[source.strip()].append(reference.strip())
                    hypotheses_dict[source.strip()] = hypothesis.strip()
                predictions = [hypotheses_dict[key] for key in hypotheses_dict]
                targets = [references_dict[key] for key in hypotheses_dict]
                # no smoothing
                smoothie = SmoothingFunction().method0
                score = np.mean([
                    sentence_bleu(target, prediction, smoothing_function=smoothie)
                    for target, prediction in zip(targets, predictions)
                ])
                return score


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, dest='references', help='File with references.')
    parser.add_argument(type=str, dest='sources', help='File with sources.')
    parser.add_argument(type=str, dest='hypotheses', help='File with hypotheses.')
    args = parser.parse_args()
    print(measure_quality(args.references, args.sources, args.hypotheses))
