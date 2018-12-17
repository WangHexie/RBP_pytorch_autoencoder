import torch
from torch.nn import Linear


class SparseNet(Linear):

    def __init__(self, in_features, out_features):
        super(SparseNet, self).__init__(in_features, out_features, bias=True)

    def forward(self, input):
        return torch.add(torch.sparse.mm(input.cuda(), self.weight.t().cuda()).cuda(), self.bias.cuda()).cuda()


if __name__ == '__main__':
    from dataset import DataSet, generate_embeddings
    import numpy as np
    import os
    import time

    data = np.load(os.path.join("usb_s", 'train_data.npz'))
    train_data = DataSet(data['train_data'], data['nums_type'])
    ebed = generate_embeddings(train_data.edge, train_data.nums_type)
    print("start")
    start = time.time()
    t = 0
    for i in train_data.next_batch(ebed, num_neg_samples=5):
        y = SparseNet(i[0][0].shape[1], 50)(i[0][0])
        # y.sum().backward()
        # print(y)
        t += 1
        if t == 100:
            break
    print(time.time() - start)
