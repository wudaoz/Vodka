from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

import dgl
from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
# from models.SGC import SGC
from dgl.nn.pytorch.conv import SGConv
from models.utils import get_training_config

from data.utils import load_tensor_data, load_ogb_data, check_writable
from data.get_dataset import get_experiment_config

from utils.logger import get_logger
from utils.metrics import accuracy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='cadets', help='Dataset')
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20, help='Label rate')
    return parser.parse_args()


def choose_path(conf):
    output_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],
                                     'cascade_random_' + str(conf['division_seed']) + '_' + str(args.labelrate))
    check_writable(output_dir)
    cascade_dir = output_dir.joinpath('cascade')
    check_writable(cascade_dir)
    return output_dir, cascade_dir


def choose_model(conf):
    conf['device'] = torch.device('cpu')
    if conf['model_name'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=features.shape[1],
            n_hidden=conf['hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=1,
            activation=F.relu,
            dropout=conf['dropout']).to(conf['device'])
    elif conf['model_name'] in ['GAT']:
        if conf['model_name'] == 'GAT':
            num_heads = 8
        else:
            num_heads = 1
        num_layers = 1
        num_out_heads = 1
        heads = ([num_heads] * num_layers) + [num_out_heads]
        model = GAT(g=G,
                    num_layers=num_layers,
                    in_dim=features.shape[1],
                    num_hidden=8,
                    num_classes=labels.max().item() + 1,
                    heads=heads,
                    activation=F.relu,
                    feat_drop=0.6,
                    attn_drop=0.6,
                    negative_slope=0.2,     # negative slope of leaky relu
                    residual=False).to(conf['device'])
    elif conf['model_name'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=features.shape[1],
                          n_hidden=conf['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=2,
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type=conf['agg_type']).to(conf['device'])

    elif conf['model_name'] == 'SGC':
        model = SGConv(in_feats=features.shape[1],
                       out_feats=labels.max().item() + 1,
                       k=2,
                       cached=True,
                       bias=False).to(conf['device'])



def train(all_logits, dur, epoch):
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    if conf['model_name'] in ['GCN']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT' ]:
        logits, _ = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])

    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[idx_train], labels[idx_train])
    acc_train = accuracy(logp[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    dur.append(time.time() - t0)
    model.eval()
    if conf['model_name'] in ['GCN']:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT']:
        logits, _ = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])

    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)
    # we save the logits for visualization later
    all_logits.append(logp.cpu().detach().numpy())
    loss_val = F.nll_loss(logp[idx_val], labels[idx_val])
    acc_val = accuracy(logp[idx_val], labels[idx_val])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    print('Epoch %d | Loss: %.4f | loss_val: %.4f | acc_train: %.4f | acc_val: %.4f | acc_test: %.4f | Time(s) %.4f' % (
        epoch, loss.item(), loss_val.item(), acc_train.item(), acc_val.item(), acc_test.item(), dur[-1]))
    return acc_val, loss_val


def model_train(conf, model, optimizer, all_logits):
    dur = []
    best = 0
    cnt = 0
    epoch = 1
    while epoch < conf['max_epoch']:
        acc_val, loss_val = train(all_logits, dur, epoch)
        epoch += 1
        if acc_val >= best:
            best = acc_val
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            cnt = 0
        else:
            cnt += 1
        if cnt == conf['patience'] or epoch == conf['max_epoch']:
            print("Stop!!!")
            # print("Saving cascade info...")
            # all_logits = all_logits[: -cnt]
            # if cascade_dir is not None:
            #     for i in range(len(all_logits)):
            #         np.savetxt(cascade_dir.joinpath(str(i) + '.txt'),
            #                    np.exp(all_logits[i]),
            #                    fmt='%.4f', delimiter='\t')
            break

    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(np.sum(dur)))


def test(conf):
    model.eval()
    if conf['model_name'] in ['GCN']]:
        logits = model(G.ndata['feat'])
    elif conf['model_name'] in ['GAT']:
        logits, G.edata['a'] = model(G.ndata['feat'])
    elif conf['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(G, G.ndata['feat'])

    else:
        raise ValueError(f'Undefined Model')
    logp = F.log_softmax(logits, dim=1)
    loss_test = F.nll_loss(logp[idx_test], labels[idx_test])
    acc_test = accuracy(logp[idx_test], labels[idx_test])
    print("Test set results: loss= {:.4f} acc_test= {:.4f}".format(
        loss_test.item(), acc_test.item()))
    return acc_test, logp


if __name__ == '__main__':
    args = arg_parse(argparse.ArgumentParser())
    config_path = Path.cwd().joinpath('models', 'train.conf.yaml')
    conf = get_training_config(config_path, model_name=args.teacher)
    config_data_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')
    conf['division_seed'] = get_experiment_config(config_data_path)['seed']
    # if args.device > 0:
    #     conf['device'] = torch.device("cuda:" + str(args.device))
    # else:
    #     conf['device'] = torch.device("cpu")
    conf['device'] = torch.device("cpu")

    conf = dict(conf, **args.__dict__)
    print(conf)
    output_dir, cascade_dir = choose_path(conf)
    logger = get_logger(output_dir.joinpath('log'))
    print(output_dir)
    print(cascade_dir)
    # random seed
    np.random.seed(conf['seed'])
    torch.manual_seed(conf['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Load data
    if conf['dataset'] in ['arxiv']:
        G, features, labels, idx_train, idx_val, idx_test = load_ogb_data(conf['dataset'], conf['device'])
    else:
        adj, adj_sp, features, labels, labels_one_hot, idx_train, idx_val, idx_test = \
            load_tensor_data(conf['model_name'], conf['dataset'], args.labelrate, conf['device'])

        print('Dataset: ', conf['dataset'])
        print('Feature shape: ', features.shape)
        print('Label shape: ', labels.shape)
        print('Number of training nodes: ', len(idx_train))
        print('Number of validation nodes: ', len(idx_val))
        print('Number of test nodes: ', len(idx_test))
        print('Number of classes: ', labels.max().item() + 1)
        
        adj_sp_row = torch.LongTensor(adj_sp.row).to('cpu')
        adj_sp_col = torch.LongTensor(adj_sp.col).to('cpu')
        G = dgl.graph((adj_sp_row, adj_sp_col)).to('cpu')

        G.ndata['feat'] = features
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    # The first layer transforms input features of size of 5 to a hidden size of 5.
    # The second layer transforms the hidden layer and produces output features of
    # size 2, corresponding to the two groups of the karate club.
    model = choose_model(conf)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=conf['learning_rate'],
                               weight_decay=conf['weight_decay'])
    all_logits = []
    model_train(conf, model, optimizer, all_logits)
    acc_test, logp = test(conf)
    preds = logp.max(1)[1].type_as(labels).cpu().numpy()
    labels = labels.cpu().numpy()
    output = np.exp(logp.cpu().detach().numpy())
    acc_test = acc_test.cpu().item()
    np.savetxt(output_dir.joinpath('preds.txt'), preds, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('labels.txt'), labels, fmt='%d', delimiter='\t')
    np.savetxt(output_dir.joinpath('output.txt'), output, fmt='%.4f', delimiter='\t')
    np.savetxt(output_dir.joinpath('test_acc.txt'), np.array([acc_test]), fmt='%.4f', delimiter='\t')
    if 'a' in G.edata:
        print('Saving Attention...')
        edge = torch.stack((G.edges()[0], G.edges()[1]),0)
        sp_att = sp.coo_matrix((G.edata['a'].cpu().detach().numpy(), edge.cpu()), shape=adj.cpu().size())
        sp.save_npz(output_dir.joinpath('attention_weight.npz'), sp_att, compressed=True)
