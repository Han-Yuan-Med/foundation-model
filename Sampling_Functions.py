import numpy as np
from sklearn.model_selection import KFold


def create_sampled_dataset(train_csv, random_seed, dataset_path, sample_times):
    for sample_id in range(sample_times):

        sample_selected = np.array([])
        sample_left = range(len(train_csv))
        sample_num = round(len(train_csv) * 0.05)

        for idx in range(1, 21):
            if idx == 10:
                train_csv.to_csv(f"{dataset_path}\\train_id_{sample_id}_{idx / 10}.csv", index=False)
                break

            np.random.seed(random_seed*sample_id)
            sample_tmp = np.random.choice(sample_left, sample_num, replace=False)
            sample_selected = np.append(sample_selected, sample_tmp)
            sample_left = np.setdiff1d(sample_left, sample_tmp)
            assert len(sample_selected) == idx * sample_num
            assert len(sample_selected) == len(np.unique(sample_selected))
            train_csv.iloc[sample_selected].to_csv(f"{dataset_path}\\train_id_{sample_id}_{idx/10}.csv", index=False)


def create_sampled_instance(train_csv, random_seed, dataset_path, sample_times):

    for sample_id in range(sample_times):
        sample_selected = np.array([])
        sample_left = range(len(train_csv))

        for idx in range(10, 101, 10):
            np.random.seed(random_seed*sample_id)
            sample_tmp = np.random.choice(sample_left, 10, replace=False)
            sample_selected = np.append(sample_selected, sample_tmp)
            sample_left = np.setdiff1d(sample_left, sample_tmp)
            assert len(sample_selected) == idx
            assert len(sample_selected) == len(np.unique(sample_selected))
            train_csv.iloc[sample_selected].to_csv(f"{dataset_path}\\train_id_{sample_id}_{idx}.csv", index=False)


def create_sampled_instance_seg(train_csv, random_seed, dataset_path, sample_times):

    for sample_id in range(sample_times):
        sample_selected = np.array([])
        sample_left = range(len(train_csv))

        for idx in range(10, 101, 10):
            np.random.seed(random_seed*sample_id)
            sample_tmp = np.random.choice(sample_left, 10, replace=False)
            sample_selected = np.append(sample_selected, sample_tmp)
            sample_left = np.setdiff1d(sample_left, sample_tmp)
            assert len(sample_selected) == idx
            assert len(sample_selected) == len(np.unique(sample_selected))
            train_csv.iloc[sample_selected].to_csv(f"{dataset_path}\\train_seg_id_{sample_id}_{idx}.csv", index=False)


def sample_dataset(train_csv, random_seed, dataset_path, sample_times):
    for sample_id in range(sample_times):
        for idx in range(10, 101, 10):
            np.random.seed(random_seed * sample_id)
            sample_tmp = np.random.choice(range(len(train_csv)), idx, replace=False)
            train_csv.iloc[sample_tmp].to_csv(f"{dataset_path}\\train_id_{sample_id}_{idx}.csv", index=False)


def create_dataset_cross_validation(train_csv, random_seed, dataset_path, sample_times, n_splits):
    # Create cross validation dataset under full train_csv
    kf = KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
    for i, (train_index, valid_index) in enumerate(kf.split(range(len(train_csv)))):
        train_csv.iloc[train_index].to_csv(f"{dataset_path}\\train_id_{0}_{1.0}_{i}.csv", index=False)
        train_csv.iloc[valid_index].to_csv(f"{dataset_path}\\valid_id_{0}_{1.0}_{i}.csv", index=False)

    for sample_id in range(sample_times):
        for idx in range(1, 10):
            np.random.seed(random_seed*sample_id)
            sample_tmp = np.random.choice(range(len(train_csv)), round(len(train_csv) * idx / 10), replace=False)

            for i, (train_index, valid_index) in enumerate(kf.split(range(len(sample_tmp)))):
                train_csv.iloc[sample_tmp[train_index]].to_csv(
                    f"{dataset_path}\\train_id_{sample_id}_{idx/10}_{i}.csv", index=False)
                train_csv.iloc[sample_tmp[valid_index]].to_csv(
                    f"{dataset_path}\\valid_id_{sample_id}_{idx/10}_{i}.csv", index=False)
