import pandas as pd
import os

order = ['collar_design_labels',
 'neckline_design_labels',
 'skirt_length_labels',
 'sleeve_length_labels',
 'neck_design_labels',
 'coat_length_labels',
 'lapel_design_labels',
 'pant_length_labels']

dirs = ['fashionAI_a',
        'fashionAI_b',
        'fashionAI_c',
        'fashionAI_d']

for d in dirs:
    os.makedirs(d)

for t in order:
    df = pd.read_csv('fashionAI/{}_train.csv'.format(t),
                     names=['image','type','category'])

    df = df.sample(frac=1, random_state=666)

    step = len(df)//4

    f = dict()
    f[0] = df[0:step]
    f[1] = df[step:2*step]
    f[2] = df[2*step:3*step]
    f[3] = df[3*step:]
    f[4] = pd.read_csv('fashionAI/{}_test.csv'.format(t),
                       names=['image','type','category'])

    for i in range(4):
        l = list(range(5))
        idxs = l[:i] + l[i+1:]
        test_df = f[i]
        train_df = pd.concat([f[i] for i in idxs])
        train_df.to_csv(os.path.join(dirs[i], '{}_train.csv'.format(t)),
                        index=False,
                        header=False)
        test_df.to_csv(os.path.join(dirs[i], '{}_test.csv'.format(t)),
                       index=False,
                       header=False)
