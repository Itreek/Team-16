import pandas as pd

metric_names = {
    'idf1': 'IDF1',
    'recall': 'Rcll',
    'precision': 'Prcn',
    'num_unique_objects': 'GT',
    'num_false_positives': 'FP',
    'num_misses': 'FN',
    'num_switches': 'IDs',
    'mota': 'MOTA',
    'motp': 'MOTP'
}

def load_mot(fname):
    df = pd.read_csv(fname,skipinitialspace=True,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],engine='python')
    del df['unused']
    del df['Confidence']
    del df['Visibility']
    del df['ClassId']
    return df

print(load_mot('det.txt'))
