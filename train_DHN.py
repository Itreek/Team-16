
from DHN_model import DHN
import torch.optim as optim
from deep_loss import focal_loss
from torch.utils import data
import torch
import numpy as np
import shutil
import os
import argparse
from os.path import realpath, dirname


def train_model(args):

    # Initializing initial learning rate
    old_lr = 0.0003

    element_dim = args.element_dim
    hidden_dim = args.hidden_dim
    target_size = args.target_size
    bidirectional = args.bidirectional
    minibatch = args.batch_size
    is_train = True

    # model
    model = DHN(element_dim, hidden_dim, target_size,
                bidirectional, minibatch, is_train=True)
    model = model.train()

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=old_lr)

    starting_epoch = 0

    # data loaders
    # prepare data
    train_dataset = RealData(args.data_path, train=True)
    val_dataset = RealData(args.data_path, train=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True)

    # Run the training loop
    for epoch in range(max(0, starting_epoch), args.epochs):

        for X, target in train_dataloader:

            model = model.train()
            X = X.squeeze(0)
            target = target.squeeze(0)

            # Forward pass
            model.hidden_row = model.init_hidden(X.size(0))
            model.hidden_col = model.init_hidden(X.size(0))

            # input to model
            new_weights = model(X)

            # clear the gradien
            model.zero_grad()

            # credit the assignment
            loss.backward()

            # update the model weights
            optimizer.step()

            # adjust learning weight
            old_lr = adjust_learning_rate(optimizer, iteration, old_lr)

            # evaluate the model

            if iteration % args.print_test == 0:
                model = model.eval()
                test_loss = []
                acc = []
                test_j = []
                test_p = []
                test_r = []

                for test_num, (data, target) in enumerate(val_dataloader):
                    data = data.squeeze(0)
                    target = target.squeeze(0)
                    if test_num == 50:
                        break

                    # after each sequence/matrix X, we init new hidden states
                    model.hidden_row = model.init_hidden(data.size(0))
                    model.hidden_col = model.init_hidden(data.size(0))

                    # input to model
                    new_weights = model(data)

                    num_positive = target.data.view(
                        target.size(0), -1).sum(dim=1).unsqueeze(1)

                    wn_val = num_positive.float() / (target.size(1) * target.size(2))

                    # case all zeros
                    wn_val.masked_fill_((wn_val == 0), 1.0)

                    # case all ones
                    wn_val.masked_fill_((wn_val == 1), 0.0)

                    weight = torch.cat(
                        [wn_val, 1.0 - wn_val], dim=1)

                    weight = weight.view(-1, 2, 1, 1).contiguous()

                    loss = 10.0 * \
                        focal_loss(new_weights, target.float(), weights=weight)

                    test_loss.append(float(loss.item()))

                    predicted = get_predicted(
                        new_weights.float().detach(), target.float().detach(), weight.detach())

                    tp = torch.sum((predicted == target.float().detach())[
                                   target.data == 1.0]).double()
                    fp = torch.sum((predicted != target.float().detach())[
                                   predicted.data == 1.0]).double()
                    fn = torch.sum((predicted != target.float().detach())[
                                   predicted.data == 0.0]).double()

                    p = tp / (tp + fp + 1e-9)
                    r = tp / (tp + fn + 1e-9)
                    test_p.append(p.item())
                    test_r.append(r.item())

                print('Epoch: [{}][{}/{}]\tLoss {:.4f}\tweighted Accuracy {:.2f} %'.format(epoch, iteration % len(train_dataloader.dataset),
                                                                                           len(
                                                                                               train_dataloader.dataset),
                                                                                           np.mean(
                                                                                               np.array(test_loss)),
                                                                                           100.0*np.mean(np.array(acc))))
                is_best = False
            iteration += 1


def adjust_learning_rate(optimizer, iteration, lr):
    lr = lr * (0.95 ** (iteration // 30000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_predicted(score, target, weight, th=0.5):

    predicted = np.zeros_like(score).cuda().float()

    for b in range(score.size(0)):
        for h in range(score.size(1)):
            value, indice = score[b, h].max(0)
            if float(value) > th:
                predicted[b, h, int(indice)] = 1.0

        num_positive = float(target[b, :, :].sum())
        num_negative = float(target.size(1)*target.size(2) - num_positive)

        num_tp = float(((predicted[b, :, :] == target[b, :, :]).float(
        ) + (target[b, :, :] == 1.0).float()).eq(2).sum())
        num_tn = float(((predicted[b, :, :] == target[b, :, :]).float(
        ) + (target[b, :, :] == 0.0).float()).eq(2).sum())

    return predicted


def prepare_Data(data_pth):

    data = list()
    d_pth = os.path.join(data_pth, 'train/')
    dirs = os.listdir(d_pth)
    for dir in dirs:
        pth = os.path.join(d_pth, dir)
        files = os.listdir(pth)
        for file in files:
            if '_m.npy' in file:
                data.append(os.path.join(pth, file))
    return data


class RealData(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_path, train=True):
        'Initialization'
        self.data_pth = data_path
        self.data = prepare_Data(data_path, train)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        m_pth = self.data[index]  # input matrix name
        split = m_pth.split('_')
        t_pth = '_'.join(split[:-1])+'_t.npy'

        # Load data and get label
        matrix = np.load(m_pth)
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))

        # random permutations
        # vertical
        if torch.rand(1).item() < 0.5:
            idx = torch.randperm(matrix.shape[1])
            matrix = matrix[0:, idx, 0:]
            target = target[0:, idx, 0:]

        # horizontal
        if torch.rand(1).item() < 0.5:
            idx = torch.randperm(matrix.shape[2])
            matrix = matrix[0:, 0:, idx]
            target = target[0:, 0:, idx]

        return [matrix, target]


if __name__ == '__main__':
    # parameters #

    curr_path = realpath(dirname(/deepmot/train_DHN/DHN_data/train))
    parser = argparse.ArgumentParser(description='train DHN')

    # data configs
    parser.add_argument('--data_path', dest='data_path',
                        default=os.path.join(curr_path, 'DHN_data/'), help='dataset root path')

    # BiRNN configs
    parser.add_argument('--element_dim', dest='element_dim',
                        default=1, type=int, help='element_dim')
    parser.add_argument('--hidden_dim', dest='hidden_dim',
                        default=256, type=int, help='hidden_dim')
    parser.add_argument('--target_size', dest='target_size',
                        default=1, type=int, help='target_size')
    parser.add_argument('--batch_size', dest='batch_size',
                        default=1, type=int, help='batch_size')
    parser.add_argument('--bidirectional',
                        action='store_true', help="Bi-RNN if set.")

    # train configs
    parser.add_argument('-b', dest='batch_size', default=1,
                        type=int, help='batch size')
    parser.add_argument('--epochs', dest='epochs', default=20,
                        type=int, help='number of training epochs')
    parser.add_argument('--print_test', dest='print_test',
                        default=20, type=int, help='test frequency')
    parser.add_argument('--print_train', dest='print_train',
                        default=10, type=int, help='training print frequency')
    parser.add_argument('--save_name', dest='save_name',
                        default='DHN', help='save folder name')

    args = parser.parse_args()

    train_model(args)
