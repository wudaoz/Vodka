import argparse
import yaml
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from data.make_dataset import get_dataset_and_split_planetoid, get_dataset, get_train_val_test_split


def get_experiment_config(config_path):
    with open(config_path, 'r') as conf:
        return yaml.load(conf, Loader=yaml.FullLoader)


def generate_data_path(dataset, dataset_source):
    if dataset_source == 'planetoid':
        return 'data/planetoid'
    elif dataset_source == 'npz':
        return 'data/npz/' + dataset + '.npz'
    else:
        print(dataset_source)
        raise ValueError(f'The "dataset_source" must be set to "planetoid" or "npz"')


# def load_dataset_and_split(labelrate, dataset):
#     # _config = get_experiment_config(config_file)
#     _config = {
#         'dataset_source': 'npz',
#         'seed': 0,
#         'train_config': {
#             'split': {
#                 'train_examples_per_class': labelrate,  # 20
#                 'val_examples_per_class': 30
#             },
#             'standardize_graph': True
#         }
#     }
#     print('_config', _config)
#     _config['data_path'] = generate_data_path(dataset, _config['dataset_source'])
#     if _config['dataset_source'] == 'planetoid':
#         return get_dataset_and_split_planetoid(dataset, _config['data_path'])
#     else:
#         adj, features, labels = get_dataset(dataset, _config['data_path'],
#                                             _config['train_config']['standardize_graph'],
#                                             _config['train_config']['split']['train_examples_per_class'],
#                                             _config['train_config']['split']['val_examples_per_class'])
#         random_state = np.random.RandomState(_config['seed'])
#         idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels,
#                                                                 **_config['train_config']['split'])
#         return adj, features, labels, idx_train, idx_val, idx_test

# from sklearn.model_selection import train_test_split

def load_dataset_and_split(labelrate, dataset):
    # _config = get_experiment_config(config_file)
    _config = {
        'dataset_source': 'planetoid',
        'seed': 0,
        'train_config': {
            'split': {
                'train_ratio': 0.8,  # 每类标签60%分给训练集
                'val_ratio': 0.1     # 每类标签20%分给验证集，剩下20%给测试集
            },
            'standardize_graph': True
        }
    }
    print('_config', _config)
    _config['data_path'] = generate_data_path(dataset, _config['dataset_source'])
    
    # 加载数据集
    if _config['dataset_source'] == 'planetoid':
        return get_dataset_and_split_planetoid(dataset, _config['data_path'])
    else:
        adj, features, labels = get_dataset(dataset, _config['data_path'],
                                            _config['train_config']['standardize_graph'])

        # 使用随机种子保证结果可重复
        random_state = np.random.RandomState(_config['seed'])
        
        # 按比例划分训练集、验证集和测试集
        idx_train, idx_val, idx_test = get_train_val_test_split_by_ratio(random_state, labels,
                                                                         _config['train_config']['split']['train_ratio'],
                                                                         _config['train_config']['split']['val_ratio'])
        return adj, features, labels, idx_train, idx_val, idx_test

def get_train_val_test_split_by_ratio(random_state, labels, train_ratio, val_ratio):
    """
    按比例划分训练集、验证集、测试集。
    参数：
    - random_state: 随机种子，用于保证可重复性
    - labels: 标签数组
    - train_ratio: 训练集比例
    - val_ratio: 验证集比例

    返回：
    - idx_train: 训练集索引
    - idx_val: 验证集索引
    - idx_test: 测试集索引
    """
    # 获取所有样本的索引
    indices = np.arange(labels.shape[0])
    
    # 首先按比例划分训练集和剩余部分（验证集 + 测试集）
    idx_train, idx_rest = train_test_split(indices, train_size=train_ratio, random_state=random_state, stratify=labels)

    # 剩下的部分按比例继续划分验证集和测试集
    val_test_ratio = val_ratio / (1 - train_ratio)  # 将剩下的部分按比例分成验证集和测试集
    idx_val, idx_test = train_test_split(idx_rest, test_size=1-val_test_ratio, random_state=random_state, stratify=labels[idx_rest])

    return idx_train, idx_val, idx_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        default='dataset.conf.yaml',
                        help='Path to the YAML configuration file for the experiment.')
    args = parser.parse_args()
    adj, features, labels, idx_train, idx_val, idx_test = load_dataset_and_split(args.config_file)
