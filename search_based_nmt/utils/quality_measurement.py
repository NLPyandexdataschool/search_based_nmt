import argparse
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import defaultdict


METHODS = {
    0: SmoothingFunction().method0,
    1: SmoothingFunction().method1,
    2: SmoothingFunction().method2,
    3: SmoothingFunction().method3,
    4: SmoothingFunction().method4,
    5: SmoothingFunction().method5,
    6: SmoothingFunction().method6,
    7: SmoothingFunction().method7
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', type=str, required=True,
                        dest='references', help='File with references.')
    parser.add_argument('--sources', type=str, required=True,
                        dest='sources', help='File with sources.')
    parser.add_argument('--hypotheses', type=str, required=True,
                        dest='hypotheses', help='File with hypotheses.')
    parser.add_argument('--n', type=int, default=0,
                        dest='n', help='Number of smoothing method (from 0 to 7)')
    return parser.parse_args()


def measure_quality(references_file_name, sources_file_name, hypotheses_file_name, n=0):
    with open(references_file_name, encoding='utf8') as references_handler,\
         open(sources_file_name, encoding='utf8') as sources_handler,\
         open(hypotheses_file_name, encoding='utf8') as hypotheses_handler:
                references_dict = defaultdict(list)
                hypotheses_dict = {}
                for reference, source, hypothesis in zip(
                        references_handler,
                        sources_handler,
                        hypotheses_handler
                ):
                    references_dict[source.strip()].append(reference.strip())
                    hypotheses_dict[source.strip()] = hypothesis.strip()

                predictions = []
                targets = []
                for key in hypotheses_dict:
                    predictions.append(hypotheses_dict[key])
                    targets.append(references_dict[key])

                smoothie = METHODS[n]
                score = corpus_bleu(
                    list_of_references=targets,
                    hypotheses=predictions,
                    smoothing_function=smoothie,
                    emulate_multibleu=True
                )
                return score


if __name__ == '__main__':
    args = parse_args()

    print(measure_quality(
        references_file_name=args.references,
        sources_file_name=args.sources,
        hypotheses_file_name=args.hypotheses,
        n=args.n
    ))
