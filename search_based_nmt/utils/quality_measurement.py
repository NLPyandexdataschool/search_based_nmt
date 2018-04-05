import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import warnings
from collections import defaultdict


def measure_quality(references_file_name, sources_file_name, hypotheses_file_name, method_number):
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
        smoothing_functions_dict = {
            0: SmoothingFunction().method0,
            1: SmoothingFunction().method1,
            2: SmoothingFunction().method2,
            3: SmoothingFunction().method3,
            4: SmoothingFunction().method4,
            5: SmoothingFunction().method5,
            6: SmoothingFunction().method6,
            7: SmoothingFunction().method7
        }
        smoothie = smoothing_functions_dict[method_number]
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
    parser.add_argument(type=int, nargs='?', dest='smoothing_method', help='Smoothing method number [0-7].', default=0)
    args = parser.parse_args()
    if args.smoothing_method not in range(8):
        raise Exception("Wrong smoothing method!")
    print(measure_quality(args.references, args.sources, args.hypotheses, args.smoothing_method))
