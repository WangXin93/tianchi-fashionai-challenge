# Merge multiple csv answer file into one

import pandas as pd
import os
import argparse

#files = ['questions_b/answer.csv',
#         'questions_b/answer_a.csv',
#         'questions_b/answer_b.csv',
#         'questions_b/answer_c.csv',
#         'questions_b/answer_d.csv']

#files = ['./questions_b/merged1.csv', './questions_b/merged2.csv']

def merge_scores(cols):
    """Merge multi str formatted scores into one str

    Used for pandas.DataFrame.apply like:

        merged = df[['col1', 'col2']].apply(merge_scores, axis=1)
    
    Args:
        cols: List of strs.
    """
    out = []
    scores = []
    for c in cols:
        scores.append([float(i) for i in c.split(';')])
    length = len(scores[0])
    for i in range(length):
        mean = sum([s[i] for s in scores]) / len(cols)
        mean = '{:.4f}'.format(mean)
        out.append(mean)
    return ';'.join(out)


def merge_csv(files, target='./questions_b/merged.csv'):
    """Merge multiple csv file into one file

    Args:
        files: List of file paths
        target: Path of merged csv file
    """
    assert len(files) > 0, "Should be at least one file"
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError('{} file not found'.format(f))

    dfs = [pd.read_csv(f, names=['image', 'type', 'scores']) for f in files]

    scores = [df['scores'] for df in dfs]
    scores = pd.concat(scores, axis=1)

    merged_score = scores.apply(merge_scores, axis=1)

    answer = dfs[0]
    answer['scores'] = merged_score

    answer.to_csv(target, header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='Answer files to be merges')
    parser.add_argument('--target', type=str, required=True, 
                        help='The path of merges answer file')
    args = parser.parse_args()
    print('Merging...')
    for i in args.files:
        print(i)
    merge_csv(args.files, args.target)
    print('Successfully merge files to {}'.format(args.target))
