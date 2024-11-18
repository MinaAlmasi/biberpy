#!/usr/bin/env python
# Simplified script to perform factor analysis and output loadings without pandas

import sys
import numpy as np
from sklearn.decomposition import FactorAnalysis

def stream_entries(file_path: str):
    with open(file_path) as f:
        for line in f:
            yield line.strip().split()

def main():
    # params
    kfactors = 5   # n of factors
    topN = 20      # n of top features per factor

    # load data from 'brown-biber.dat' (assuming this is your data matrix)
    data_file = 'dailydialog-completions.csv'
    print(f'Loading data from {data_file}')
    data_gen = stream_entries(data_file)

    # read headers
    headers = next(data_gen)
    num_features = len(headers) - 1  # Assuming first column is an ID or label

    # init lists to store data
    data_list = []
    feature_names = headers[1:]  # Skip the first column if it's an ID

    # read data into a list
    for line in data_gen:
        # skip empty lines
        if not line:
            continue
        # convert values to float, skip the first column (ID)
        try:
            row = [float(value) for value in line[1:]]
            data_list.append(row)
        except ValueError:
            # skip rows with non-numeric data
            continue

    # data to numpy array
    data_matrix = np.array(data_list)

    # if we have enough variables for the desired number of factors
    if data_matrix.shape[1] < kfactors:
        print("Not enough numerical features to extract the desired number of factors.")
        sys.exit(1)

    # factor analysis
    factor = FactorAnalysis(n_components=kfactors)
    factor.fit(data_matrix)
    loadings = factor.components_.T  # Variables x Factors

    # the loadings
    with open('loadings.txt', 'w') as f:
        f.write('# Loadings for factors\n')
        for i in range(kfactors):
            print(f'Processing Factor {i+1}')
            factor_loadings = loadings[:, i]
            # Create a list of tuples (feature_name, loading_value)
            feature_loadings = list(zip(feature_names, factor_loadings))
            # Sort features based on absolute value of loadings
            sorted_features = sorted(feature_loadings, key=lambda x: abs(x[1]), reverse=True)
            f.write(f'Factor{i+1}=')
            for feature, loading_value in sorted_features[:topN]:
                f.write(f'{loading_value:+1.4f}*{feature} ')
            f.write('\n')

if __name__ == '__main__':
    main()