import os
import random
import argparse
import torch
import numpy as np

import graph_net
import utils
import trainer


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
parser.add_argument('--data_root', type=str, default='/data/office31', help='path to dataset root')
parser.add_argument('--image_list_root', type=str, default='data/office', help='path to image list')
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

    # train on source domain and compute domain inheritability
    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders)

    # find the maximum inheritability domain
    temp_test_loaders = dict(dset_loaders['target_test'])
    max_inherit_domain = trainer.compute_domain_inheritability(config, base_network, classifier_gnn, temp_test_loaders)

    # iterate over all domains
    for _ in range(config['ndomains']):
        print('Starting the adaptation...')
        ######## Step 1: train on the chosen target domain with maximum inheritance ##########
        base_network, classifier_gnn = trainer.adapt_target(config, base_network, classifier_gnn,
                                                            dset_loaders, max_inherit_domain)

        ######### Step 2: obtain the target pseudo labels and upgrade source domain ##########
        trainer.upgrade_source_domain(config, max_inherit_domain, dsets,
                                      dset_loaders, base_network, classifier_gnn)

        ######### Step 3: recompute model inheritability ###########
        # remove already considered domain
        del temp_test_loaders[max_inherit_domain]
        # find the maximum inheritability domain
        if len(temp_test_loaders.keys()) > 0:
            print(temp_test_loaders.keys())
            max_inherit_domain = trainer.compute_domain_inheritability(config, base_network,
                                                                       classifier_gnn, temp_test_loaders)
    ######### Step 4: fine-tuning stage ###########
    config['source_iters'] = config['finetune_iters']
    base_network, classifier_gnn = trainer.train_source(config, base_network, classifier_gnn, dset_loaders)
    print('Finished training and evaluation!')

    # save models
    if args.save_models:
        torch.save(base_network.cpu().state_dict(), os.path.join(config['output_path'], 'base_network.pth.tar'))
        torch.save(classifier_gnn.cpu().state_dict(), os.path.join(config['output_path'], 'classifier_gnn.pth.tar'))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

