from argparse import ArgumentParser
import os
import shutil
import glob


TO_DELETE = [
    'checkpoint',
    'flags.txt',
    'flags_t2t.txt',
    'graph.pbtxt',
    'hparams.json'
]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--skip', nargs='*')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.skip is None:
        args.skip = []

    if args.dir.endswith('/'):
        args.dir = args.dir[:-1]

    listdir = os.listdir(args.dir)

    for n in TO_DELETE:
        if n not in args.skip:
            if n in listdir:
                path = os.path.join(args.dir, n)
                try:
                    os.remove(path)
                except OSError as e:
                    print ('Exception while removing {}'.format(path), e)

    if 'eval_one_pass' not in args.skip:
        if 'eval_one_pass' in listdir:
            try:
                shutil.rmtree(os.path.join(args.dir, 'eval_one_pass'))
            except OSError as e:
                print ('Exception while removing {}'.format('eval_one_pass'), e)

    if 'events' not in args.skip:
        events = glob.glob(args.dir + '/events*')
        for ev in events:
            try:
                os.remove(ev)
            except Exception as e:
                print ('Exception while removing {}'.format(ev), e)

    if 'models' not in args.skip:
        models = glob.glob(args.dir + '/model.ckpt*')
        for m in models:
            try:
                os.remove(m)
            except Exception as e:
                print ('Exception while removing {}'.format(m), e)


if __name__ == '__main__':
    main()
