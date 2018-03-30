import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import warnings


def measure_quality(predicted_file_name, answers_file_name):
    with open(predicted_file_name) as predict_handler, open(answers_file_name) as answers_handler:
        predictions = [line[:-1].split() for line in predict_handler]
        answers = [line[:-1] for line in answers_handler]
        if len(predictions) != len(answers):
            raise Exception('Input files must have same length!')
        # no smoothing
        smoothie = SmoothingFunction().method0
        score = np.mean([
            sentence_bleu(answers, prediction, smoothing_function=smoothie)
            for answer, prediction in zip(answers, predictions)
        ])
    return score


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, nargs=2, dest='paths')
    args = parser.parse_args()
    print(measure_quality(*args.paths))
