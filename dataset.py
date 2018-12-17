import collections
import copy
import os
from collections import namedtuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
from torch.sparse import FloatTensor
import torch


Datasets = collections.namedtuple('Datasets', ['train', 'test', 'embeddings', 'node_cluster',
                                               'labels', 'idx_label', 'label_name'])


def add_attr_to_nametuple(nt):
    NTB = namedtuple("SparseTensorValue", ["indices", "values", "dense_shape", "ndim", "shape"])
    a = NTB(nt.indices, nt.values, nt.dense_shape, len(nt.dense_shape), nt.dense_shape)
    return a


class DataSet(object):
    time_used = 0

    def __init__(self, edge, nums_type, **kwargs):
        self.edge = edge
        np.random.shuffle(self.edge)
        self.edge_set = set(map(tuple, edge))  ### ugly code, need to be fixed
        self.nums_type = nums_type
        self.kwargs = kwargs
        self.nums_examples = len(edge)
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, embeddings, batch_size=16, num_neg_samples=1, pair_radio=0.9, sparse_input=False):
        """
            Return the next `batch_size` examples from this data set.
            if num_neg_samples = 0, there is no negative sampling.
        """
        while 1:
            import time
            start_time = time.time()
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
            if self.index_in_epoch > self.nums_examples:
                self.epochs_completed += 1
                np.random.shuffle(self.edge)
                start = 0
                self.index_in_epoch = batch_size
                assert self.index_in_epoch <= self.nums_examples
            end = self.index_in_epoch
            neg_data = []

            for i in range(start, end):
                ### warning !!! we need deepcopy to copy list
                index = copy.deepcopy(self.edge[i])
                n_neg = 0
                while (n_neg < num_neg_samples):
                    mode = np.random.rand()
                    if mode < pair_radio:
                        type_ = np.random.randint(3)
                        node = np.random.randint(self.nums_type[type_])
                        index[type_] = node
                    else:
                        types_ = np.random.choice(3, 2, replace=False)
                        node_1 = np.random.randint(self.nums_type[types_[0]])
                        node_2 = np.random.randint(self.nums_type[types_[1]])
                        index[types_[0]] = node_1
                        index[types_[1]] = node_2
                    if tuple(index) in self.edge_set:
                        continue
                    n_neg += 1
                    neg_data.append(index)
            if len(neg_data) > 0:
                batch_data = np.vstack((self.edge[start:end], neg_data))
                nums_batch = len(batch_data)
                labels = np.zeros(nums_batch)
                labels[0:end - start] = 1
                perm = np.random.permutation(nums_batch)
                batch_data = batch_data[perm]
                labels = labels[perm]
            else:
                batch_data = self.edge[start:end]
                nums_batch = len(batch_data)
                labels = np.ones(len(batch_data))
            batch_e = embedding_lookup(embeddings, batch_data, sparse_input)
            finish_time = time.time()
            DataSet.time_used = DataSet.time_used + finish_time - start_time
            yield [batch_e[i] for i in range(3)], torch.FloatTensor([labels]).cuda()


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    coo.astype(dtype='float32')
    indices = np.mat([coo.row, coo.col]).transpose()

    a = FloatTensor(torch.LongTensor(indices.T.astype('int32')), torch.FloatTensor(coo.data), torch.Size(coo.shape))
    a.requires_grad=False
    return a.cuda()


def embedding_lookup(embeddings, index, sparse_input=False):
    if sparse_input:
        return [embeddings[i][index[:, i], :].todense() for i in range(3)]
    else:
        lookups = [embeddings[i][index[:, i], :] for i in range(3)]
        sparse_tensor = [convert_sparse_matrix_to_sparse_tensor(lookup) for lookup in lookups]
        return sparse_tensor


def read_data_sets(train_dir):
    TRAIN_FILE = 'train_data.npz'
    TEST_FILE = 'test_data.npz'
    data = np.load(os.path.join(train_dir, TRAIN_FILE))
    train_data = DataSet(data['train_data'], data['nums_type'])
    labels = data['labels'] if 'labels' in data else None
    idx_label = data['idx_label'] if 'idx_label' in data else None
    label_set = data['label_name'] if 'label_name' in data else None
    del data
    data = np.load(os.path.join(train_dir, TEST_FILE))
    test_data = DataSet(data['test_data'], data['nums_type'])
    node_cluster = data['node_cluster'] if 'node_cluster' in data else None
    test_labels = data['labels'] if 'labels' in data else None
    del data
    embeddings = generate_embeddings(train_data.edge, train_data.nums_type)
    return Datasets(train=train_data, test=test_data, embeddings=embeddings, node_cluster=node_cluster,
                    labels=labels, idx_label=idx_label, label_name=label_set)


def generate_H(edge, nums_type):
    nums_examples = len(edge)
    H = [csr_matrix((np.ones(nums_examples), (edge[:, i], range(nums_examples))), shape=(nums_type[i], nums_examples))
         for i in range(3)]
    return H


def dense_to_onehot(labels):
    return np.array(map(lambda x: [x * 0.5 + 0.5, x * -0.5 + 0.5], list(labels)), dtype=float)


def generate_embeddings(edge, nums_type, H=None):
    if H is None:
        H = generate_H(edge, nums_type)
    embeddings = [H[i].dot(s_vstack([H[j] for j in range(3) if j != i]).T).astype('float') for i in range(3)]
    ### 0-1 scaling
    for i in range(3):
        col_max = np.array(embeddings[i].max(0).todense()).flatten()
        _, col_index = embeddings[i].nonzero()
        embeddings[i].data /= col_max[col_index]
    return embeddings


if __name__ == '__main__':
    data = np.load(os.path.join("usb_s", 'train_data.npz'))
    train_data = DataSet(data['train_data'], data['nums_type'])
    ebed = generate_embeddings(train_data.edge, train_data.nums_type)
    print("start")
    import time

    start = time.time()
    t = 0
    for i in train_data.next_batch(ebed, num_neg_samples=5):
        t += 1
        if t == 100:
            break
    print(time.time() - start)
