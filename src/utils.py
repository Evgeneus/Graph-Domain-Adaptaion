import os
import torch
from torch.utils.data import DataLoader

import networks
import preprocess
from preprocess import ImageList


def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i += 1
    return optimizer


def build_config(args):
    config = {
        'method': args.method,
        'ndomains': 2,
        'output_path': 'results/' + args.output_dir,
        'threshold': args.threshold,
        'edge_features': args.edge_features,
        'source_iters': args.source_iters,
        'finetune_iters': args.finetune_iters,
        'adapt_iters': args.adapt_iters,
        'test_interval': args.test_interval,
        'num_workers': args.num_workers,
        'lambda_edge': args.lambda_edge,
        'lambda_node': args.lambda_node,
        'lambda_adv': args.lambda_adv,
        'random_dim': args.rand_proj,
    }
    # preprocessing params
    config['prep'] = {
        'test_10crop': False,
        'params':
            {'resize_size': 256,
             'crop_size': 224,
             },
    }
    # backbone params
    config['encoder'] = {
        'name': networks.ResNetFc,
        'params': {'resnet_name': args.encoder,
                   'use_bottleneck': True,
                   'bottleneck_dim': 256,
                   'new_cls': True,
                   },
    }
    # optimizer params
    config['optimizer'] = {
        'type': torch.optim.SGD,
        'optim_params': {
            'lr': args.lr,
             'momentum': 0.9,
             'weight_decay': args.wd,
             'nesterov': True,
             },
        'lr_type': 'inv',
        'lr_param': {
            'lr': args.lr,
            'gamma': 0.001,
            'power': 0.75,
        },
    }
    # dataset params
    config['dataset'] = args.dataset
    config['data_root'] = args.data_root
    config['data'] = {
        'image_list_root': args.image_list_root,
        'source': {
            'name': args.source,
            'batch_size': args.source_batch,
        },
        'target': {
            'name': args.target,
            'batch_size': args.target_batch,
        },
        'test': {
            'name': args.target,
            'batch_size': 512,
        },
    }
    # set number of classes
    if config['dataset'] == 'office31':
        config['encoder']['params']['class_num'] = 31
    elif config['dataset'] == 'office-home':
        config['encoder']['params']['class_num'] = 65
    elif config['dataset'] == 'domain-net':
        config['encoder']['params']['class_num'] = 345
    elif config['dataset'] == 'pacs':
        config['encoder']['params']['class_num'] = 7
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')

    # set pre-processing transformations
    config['prep'] = {
        'source': preprocess.image_train(**config['prep']['params']),
        'target': preprocess.image_train(**config['prep']['params']),
        'test': preprocess.image_test(**config["prep"]['params']),
    }
    # create output folder and log file
    if not os.path.exists(config['output_path']):
        os.system('mkdir -p '+config['output_path'])
    config['out_file'] = open(os.path.join(config['output_path'], 'log.txt'), 'w')

    # print pout config values
    config['out_file'].write(str(config)+'\n')
    config['out_file'].flush()

    return config


def build_data(config):
    dsets = {
        'target_train': {},
        'target_test': {},
    }
    dset_loaders = {
        'target_train': {},
        'target_test': {},
    }
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    target_bs = data_config["target"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    # source dataloader
    dsets['source'] = ImageList(image_root=config['data_root'], image_list_root=data_config['image_list_root'],
                                dataset=data_config['source']['name'], transform=config['prep']["source"],
                                domain_label=0, dataset_name=config['dataset'], split='train')
    dset_loaders['source'] = DataLoader(dsets['source'], batch_size=train_bs, shuffle=True,
                                        num_workers=config['num_workers'], drop_last=True, pin_memory=True)

    # target dataloader
    for dset_name in sorted(data_config['target']['name']):
        # create train and test datasets for a target domain
        dsets['target_train'][dset_name] = ImageList(image_root=config['data_root'],
                                                     image_list_root=data_config['image_list_root'],
                                                     dataset=dset_name, transform=config['prep']['target'],
                                                     domain_label=1, dataset_name=config['dataset'], split='train')
        dsets['target_test'][dset_name] = ImageList(image_root=config['data_root'],
                                                    image_list_root=data_config['image_list_root'],
                                                    dataset=dset_name, transform=config['prep']['test'],
                                                    domain_label=1, dataset_name=config['dataset'], split='test')
        # create train and test dataloaders for a target domain
        dset_loaders['target_train'][dset_name] = DataLoader(dataset=dsets['target_train'][dset_name],
                                                             batch_size=target_bs, shuffle=True,
                                                             num_workers=config['num_workers'], drop_last=True)
        dset_loaders['target_test'][dset_name] = DataLoader(dataset=dsets['target_test'][dset_name],
                                                            batch_size=test_bs, num_workers=config['num_workers'],
                                                            pin_memory=True)
    return dsets, dset_loaders
