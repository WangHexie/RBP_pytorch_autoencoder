import torch

from sparse_layer import SparseNet


class HyperEmbedding(torch.nn.Module):
    def __init__(self, input_dims: list, embedding_size, H):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(HyperEmbedding, self).__init__()
        self.encoding = [SparseNet(dim, embedding_size) for dim in input_dims]
        self.decoding = [torch.nn.Linear(embedding_size, dim).cuda() for dim in input_dims]
        self.hidden_layer = torch.nn.Linear(len(input_dims) * embedding_size, H).cuda()
        self.output = torch.nn.Linear(H, 1).cuda()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        encoded = [torch.tanh(self.encoding[i](x[i])) for i in range(len(self.encoding))]
        decoded = [torch.sigmoid(self.decoding[i](encoded[i])) for i in range(len(encoded))]

        merged = torch.cat(encoded, dim=1).cuda()
        h_relu = self.hidden_layer(merged).clamp(min=0)
        y_pred = self.output(h_relu)
        return y_pred, decoded


if __name__ == '__main__':
    from dataset import DataSet, generate_embeddings
    import numpy as np
    import os
    import time
    import torch.nn.functional as F
    from torch import optim


    def kl_divergence(p, q):
        '''
        args:
            2 tensors `p` and `q`
        returns:
            kl divergence between the softmax of `p` and `q`
        '''
        p = F.softmax(p)
        q = F.softmax(q)

        s1 = torch.sum(p * torch.log(p / q))
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
        return s1 + s2


    data = np.load(os.path.join("usb_s", 'train_data.npz'))
    train_data = DataSet(data['train_data'], data['nums_type'])
    dims = [sum(train_data.nums_type) - n for n in train_data.nums_type]
    ebed = generate_embeddings(train_data.edge, train_data.nums_type)
    print("start")
    start = time.time()
    t = 0
    net = HyperEmbedding(dims, 64, 200).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    for inputs, label in train_data.next_batch(ebed, num_neg_samples=5):
        optimizer.zero_grad()
        y, decoded = net(inputs)
        loss = -(sum(
            [torch.pow((torch.sparse.mm(inputs[i].cpu(), decoded[i].t().cpu()).cpu()), 2).mean(-1) for i in
             range(len(inputs))])).mean()
        print(loss)
        loss.backward()
        optimizer.step()

        print(loss)
        # print(y)
        t += 1
        if t == 500:
            break
    print(time.time() - start)
