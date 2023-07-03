import itertools
import pandas as pd


class DataFrameSplitter:
    def __init__(self, method="ratio"):
        self.method = method

    def train_test_split(self, dataset, labels, verbose=0, **options):
        if self.method == "trials":
            train_trials = options.get('train_trials', None)
            trial_col = options.get('trial_col', None)
        elif self.method == "ratio":
            train_ratio = options.get('train_ratio', None)
        else:
            raise ValueError("You must define the method of splitting: 'trials' or 'ratio'")

        columns = dataset.columns
        train_data = pd.DataFrame(columns=columns)
        test_data = pd.DataFrame(columns=columns)

        label_values = list()
        for label in labels:
            unique_vals = sorted(dataset[label].unique())
            label_values.append(unique_vals)
        combs_of_label_values = list(itertools.product(*label_values))

        for i, comb in enumerate(combs_of_label_values):
            seg_data = dataset.copy()
            for j, label in enumerate(labels):
                seg_data = seg_data[seg_data[label] == comb[j]]
            seg_data.reset_index(drop=True, inplace=True)

            if seg_data.shape[0] > 0:
                if self.method == "trials":
                    if seg_data[trial_col][0] in train_trials:
                        train_data = train_data.append(seg_data)
                    else:
                        test_data = test_data.append(seg_data)
                elif self.method == "ratio":
                    split_index = int(seg_data.shape[0] * train_ratio)
                    train_data = train_data.append(seg_data[:split_index])
                    test_data = test_data.append(seg_data[split_index:])

            if verbose > 2:
                print("Seg_Shape:{} | TrainData:{} | TestData:{} | {}:{} | progress:{}%.".format(
                    seg_data.shape, train_data.shape, test_data.shape, labels, comb,
                    round((i / len(combs_of_label_values)) * 100)))
            elif verbose > 1:
                print("Seg_Shape:{} | TrainData:{} | TestData:{} | {}:{} | progress:{}%.".format(
                    seg_data.shape, train_data.shape, test_data.shape, labels, comb,
                    round((i / len(combs_of_label_values)) * 100)), end="\r")
            elif verbose > 0:
                print("progress:{}%.".format(round((i / len(combs_of_label_values)) * 100)), end="\r")

        assert dataset.shape[0] == train_data.shape[0] + test_data.shape[0]
        assert dataset.shape[1] == train_data.shape[1] == test_data.shape[1]

        return train_data, test_data
