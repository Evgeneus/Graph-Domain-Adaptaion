import os
import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

import graph_net
import utils
import trainer
import networks
import preprocess


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Graph Curriculum Domain Adaptaion')
# model args
parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E'])
parser.add_argument('--encoder', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50'])
parser.add_argument('--rand_proj', type=int, default=1024, help='random projection dimension')
parser.add_argument('--edge_features', type=int, default=128, help='graph edge features dimension')
parser.add_argument('--save_models', action='store_true', help='whether to save encoder, mlp and gnn models')
# dataset args
parser.add_argument('--dataset', type=str, default='office31', choices=['office31', 'office-home', 'pacs',
                                                                        'domain-net'], help='dataset used')
parser.add_argument('--source', default='amazon', help='name of source domain')
parser.add_argument('--target', nargs='+', default=['dslr', 'webcam'], help='names of target domains')
parser.add_argument('--data_root', type=str, default='data/office31', help='path to dataset root')
# training args
parser.add_argument('--source_iters', type=int, default=100, help='number of source pre-train iters')
parser.add_argument('--adapt_iters', type=int, default=3000, help='number of iters for a curriculum adaptation')
parser.add_argument('--finetune_iters', type=int, default=1000, help='number of fine-tuning iters')
parser.add_argument('--test_interval', type=int, default=500, help='interval of two continuous test phase')
parser.add_argument('--output_dir', type=str, default='res', help='output directory')
parser.add_argument('--source_batch', type=int, default=32)
parser.add_argument('--target_batch', type=int, default=32)
# optimization args
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--lambda_edge', default=1., type=float, help='edge loss weight')
parser.add_argument('--lambda_node', default=0.3, type=float, help='node classification loss weight')
parser.add_argument('--lambda_adv', default=1.0, type=float, help='adversarial loss weight')
parser.add_argument('--threshold', type=float, default=0.7, help='threshold for pseudo labels')
parser.add_argument('--seed', type=int, default=0, help='random seed for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloaders')


def main(args):
    # fix random seed
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # create train configurations
    args.use_cgct_mask = True  # used in CGCT for pseudo label mask in target datasets
    config = utils.build_config(args)
    # prepare data
    dsets, dset_loaders = utils.build_data(config)
    # set base network
    net_config = config['encoder']
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.to(DEVICE)
    print(base_network)
    # set GNN classifier
    classifier_gnn = graph_net.ClassifierGNN(in_features=base_network.bottleneck.out_features,
                                             edge_features=config['edge_features'],
                                             nclasses=base_network.fc.out_features,
                                             device=DEVICE)
    classifier_gnn = classifier_gnn.to(DEVICE)
    print(classifier_gnn)

    # train on source domain
    log_str = '==> Step 1: Pre-training on the source dataset ...'
    utils.write_logs(config, log_str)

    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders)

    log_str = '==> Finished pre-training on source!\n'
    utils.write_logs(config, log_str)

    # create random layer and adversarial network
    class_num = config['encoder']['params']['class_num']
    random_layer = networks.RandomLayer([base_network.output_num(), class_num], config['random_dim'], DEVICE)
    adv_net = networks.AdversarialNetwork(config['random_dim'], config['random_dim'], config['ndomains'])
    random_layer = random_layer.to(DEVICE)
    adv_net = adv_net.to(DEVICE)
    print(random_layer)
    print(adv_net)

    # run adaptation episodes
    log_str = '==> Starting the adaptation'
    utils.write_logs(config, log_str)
    for curri_iter in range(len(config['data']['target']['name'])):
        ######## Step 1: train one adaptation episod on combined target domains ##########
        target_train_datasets = preprocess.ConcatDataset(dsets['target_train'].values())
        dset_loaders['target_train'] = DataLoader(dataset=target_train_datasets,
                                                  batch_size=config['data']['target']['batch_size'],
                                                  shuffle=True, num_workers=config['num_workers'],
                                                  drop_last=True)

        base_network, classifier_gnn = trainer.adapt_target_cgct(config, base_network, classifier_gnn,
                                                                 dset_loaders, random_layer, adv_net)

        log_str = '==> Finishing {} adaptation episode!\n'.format(curri_iter)
        utils.write_logs(config, log_str)

        ######### Step 2: obtain the target pseudo labels and upgrade target domains ##########
        trainer.upgrade_target_domains(config, dsets, dset_loaders, base_network, classifier_gnn, curri_iter)

    ######### Step 3: fine-tuning stage ###########
    log_str = '==> Step 3: Fine-tuning on pseudo-source dataset ...'
    utils.write_logs(config, log_str)

    config['source_iters'] = config['finetune_iters']
    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders)

    log_str = 'Finished training and evaluation!'
    utils.write_logs(config, log_str)

    # save models
    if args.save_models:
        torch.save(base_network.cpu().state_dict(), os.path.join(config['output_path'], 'base_network.pth.tar'))
        torch.save(classifier_gnn.cpu().state_dict(), os.path.join(config['output_path'], 'classifier_gnn.pth.tar'))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
