import torch
import torch.nn as nn
import networks
import transfer_loss
from preprocess import ImageList, ConcatDataset
from torch.utils.data import DataLoader
import utils
from main import DEVICE


def evaluate(i, config, base_network, classifier_gnn, target_test_dset_dict):
    base_network.eval()
    classifier_gnn.eval()
    mlp_accuracy_list, gnn_accuracy_list = [], []
    for dset_name, test_loader in target_test_dset_dict.items():
        test_res = eval_domain(config, test_loader, base_network, classifier_gnn)
        mlp_accuracy, gnn_accuracy = test_res['mlp_accuracy'], test_res['gnn_accuracy']
        mlp_accuracy_list.append(mlp_accuracy)
        gnn_accuracy_list.append(gnn_accuracy)
        # print out test accuracy for domain
        log_str = 'Dataset:%s\tTest Accuracy mlp %.4f\tTest Accuracy gnn %.4f'\
                  % (dset_name, mlp_accuracy * 100, gnn_accuracy * 100)
        config['out_file'].write(log_str + '\n')
        config['out_file'].flush()
        print(log_str)

    # print out domains averaged accuracy
    mlp_accuracy_avg = sum(mlp_accuracy_list) / len(mlp_accuracy_list)
    gnn_accuracy_avg = sum(gnn_accuracy_list) / len(gnn_accuracy_list)
    log_str = 'iter: %d, Avg Accuracy MLP Classifier: %.4f, Avg Accuracy GNN classifier: %.4f'\
              % (i, mlp_accuracy_avg * 100., gnn_accuracy_avg * 100.)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    print(log_str)
    base_network.train()
    classifier_gnn.train()


def eval_domain(config, test_loader, base_network, classifier_gnn):
    logits_mlp_all, logits_gnn_all, confidences_gnn_all, labels_all = [], [], [], []
    with torch.no_grad():
        iter_test = iter(test_loader)
        for i in range(len(test_loader)):
            data = iter_test.next()
            inputs = data['img'].to(DEVICE)
            # forward pass
            feature, logits_mlp = base_network(inputs)
            # check if number of samples is greater than 1
            if len(inputs) == 1:
                # gnn cannot handle only one sample ... use MLP instead
                # this can be encountered if len_dataset % test_batch == 1
                logits_gnn = logits_mlp
            else:
                logits_gnn, _ = classifier_gnn(feature)
            logits_mlp_all.append(logits_mlp.cpu())
            logits_gnn_all.append(logits_gnn.cpu())
            confidences_gnn_all.append(nn.Softmax(dim=1)(logits_gnn_all[-1]).max(1)[0])
            labels_all.append(data['target'])
    # concatenate data
    logits_mlp = torch.cat(logits_mlp_all, dim=0)
    logits_gnn = torch.cat(logits_gnn_all, dim=0)
    confidences_gnn = torch.cat(confidences_gnn_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    # predict class labels
    _, predict_mlp = torch.max(logits_mlp, 1)
    _, predict_gnn = torch.max(logits_gnn, 1)
    mlp_accuracy = torch.sum(predict_mlp == labels).item() / len(labels)
    gnn_accuracy = torch.sum(predict_gnn == labels).item() / len(labels)
    # compute mask for high confident samples
    sample_masks_bool = (confidences_gnn > config['threshold'])
    sample_masks_idx = torch.nonzero(sample_masks_bool, as_tuple=True)[0].numpy()
    # compute accuracy of pseudo labels
    total_pseudo_labels = len(sample_masks_idx)
    if len(sample_masks_idx) > 0:
        correct_pseudo_labels = torch.sum(predict_gnn[sample_masks_bool] == labels[sample_masks_bool]).item()
        pseudo_label_acc = correct_pseudo_labels / total_pseudo_labels
    else:
        correct_pseudo_labels = -1.
        pseudo_label_acc = -1.
    out = {
        'mlp_accuracy': mlp_accuracy,
        'gnn_accuracy': gnn_accuracy,
        'confidences_gnn': confidences_gnn,
        'pred_cls': predict_gnn.numpy(),
        'sample_masks': sample_masks_idx,
        'pseudo_label_acc': pseudo_label_acc,
        'correct_pseudo_labels': correct_pseudo_labels,
        'total_pseudo_labels': total_pseudo_labels,
    }
    return out


def select_closest_domain(config, base_network, classifier_gnn, temp_test_loaders):
    """
    This function selects the closest domain (Stage 2 in Algorithm 2 of Supp Mat) where adaptation need to be performed.
    In the code we compute the mean of the max probability of the target samples from a domain, which can be 
    considered as inversely proportional to the mean of the entropy.

    Higher the max probability == lower is the entropy == higher the inheritability/similarity
    """
    base_network.eval()
    classifier_gnn.eval()
    max_inherit_val = 0.
    for dset_name, test_loader in temp_test_loaders.items():
        test_res = eval_domain(config, test_loader, base_network, classifier_gnn)
        domain_inheritability = test_res['confidences_gnn'].mean().item()
        
        if domain_inheritability > max_inherit_val:
            max_inherit_val = domain_inheritability
            max_inherit_domain_name = dset_name

    print('Most similar target domain: %s' % (max_inherit_domain_name))
    log_str = 'Most similar target domain: %s' % (max_inherit_domain_name)
    config['out_file'].write(log_str + '\n')
    config['out_file'].flush()
    return max_inherit_domain_name


def train_source(config, base_network, classifier_gnn, dset_loaders):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() +\
                     [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))

    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    base_network.train()
    classifier_gnn.train()
    len_train_source = len(dset_loaders["source"])
    for i in range(config['source_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        batch_source = iter_source.next()
        inputs_source, labels_source = batch_source['img'].to(DEVICE), batch_source['target'].to(DEVICE)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp = base_network(inputs_source)
        mlp_loss = ce_criterion(logits_mlp, labels_source)

        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features_source)
        gnn_loss = ce_criterion(logits_gnn, labels_source)
        # compute edge loss
        edge_gt, edge_mask = classifier_gnn.label2edge(labels_source.unsqueeze(dim=0))
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

        # total loss and backpropagation
        loss = mlp_loss + config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()

        # printout train loss
        if i % 20 == 0 or i == config['source_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss:%.4f\tGNN loss:%.4f\tEdge loss:%.4f' % (i,
                  config['source_iters'], mlp_loss.item(), gnn_loss.item(), edge_loss.item())
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'])

    return base_network, classifier_gnn


def adapt_target(config, base_network, classifier_gnn, dset_loaders, max_inherit_domain):
    # define loss functions
    criterion_gedge = nn.BCELoss(reduction='mean')
    ce_criterion = nn.CrossEntropyLoss()
    # add random layer and adversarial network
    class_num = config['encoder']['params']['class_num']
    random_layer = networks.RandomLayer([base_network.output_num(), class_num], config['random_dim'], DEVICE)
    
    adv_net = networks.AdversarialNetwork(config['random_dim'], config['random_dim'], config['ndomains'])
    
    random_layer.to(DEVICE)
    adv_net = adv_net.to(DEVICE)

    # configure optimizer
    optimizer_config = config['optimizer']
    parameter_list = base_network.get_parameters() + adv_net.get_parameters() \
                     + [{'params': classifier_gnn.parameters(), 'lr_mult': 10, 'decay_mult': 2}]
    optimizer = optimizer_config['type'](parameter_list, **(optimizer_config['optim_params']))
    # configure learning rates
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group['lr'])
    schedule_param = optimizer_config['lr_param']

    # start train loop
    len_train_source = len(dset_loaders['source'])
    len_train_target = len(dset_loaders['target_train'][max_inherit_domain])
    # set nets in train mode
    base_network.train()
    classifier_gnn.train()
    adv_net.train()
    random_layer.train()
    for i in range(config['adapt_iters']):
        optimizer = utils.inv_lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        # get input data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders['source'])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders['target_train'][max_inherit_domain])
        batch_source = iter_source.next()
        batch_target = iter_target.next()
        inputs_source, inputs_target = batch_source['img'].to(DEVICE), batch_target['img'].to(DEVICE)
        labels_source = batch_source['target'].to(DEVICE)
        domain_source, domain_target = batch_source['domain'].to(DEVICE), batch_target['domain'].to(DEVICE)
        domain_input = torch.cat([domain_source, domain_target], dim=0)

        # make forward pass for encoder and mlp head
        features_source, logits_mlp_source = base_network(inputs_source)
        features_target, logits_mlp_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        logits_mlp = torch.cat((logits_mlp_source, logits_mlp_target), dim=0)
        softmax_mlp = nn.Softmax(dim=1)(logits_mlp)
        mlp_loss = ce_criterion(logits_mlp_source, labels_source)

        # *** GNN at work ***
        # make forward pass for gnn head
        logits_gnn, edge_sim = classifier_gnn(features)
        gnn_loss = ce_criterion(logits_gnn[:labels_source.size(0)], labels_source)
        # compute pseudo-labels for affinity matrix by mlp classifier
        out_target_class = torch.softmax(logits_mlp_target, dim=1)
        target_score, target_pseudo_labels = out_target_class.max(1, keepdim=True)
        idx_pseudo = target_score > config['threshold']
        target_pseudo_labels[~idx_pseudo] = classifier_gnn.mask_val
        # combine source labels and target pseudo labels for edge_net
        node_labels = torch.cat((labels_source, target_pseudo_labels.squeeze(dim=1)), dim=0).unsqueeze(dim=0)
        # compute source-target mask and ground truth for edge_net
        edge_gt, edge_mask = classifier_gnn.label2edge(node_labels)
        # compute edge loss
        edge_loss = criterion_gedge(edge_sim.masked_select(edge_mask), edge_gt.masked_select(edge_mask))

        # *** Adversarial net at work ***
        if config['method'] == 'CDAN+E':
            entropy = transfer_loss.Entropy(softmax_mlp)
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp], adv_net,
                                            entropy, networks.calc_coeff(i), random_layer, domain_input)
        elif config['method'] == 'CDAN':
            trans_loss = transfer_loss.CDAN(config['ndomains'], [features, softmax_mlp],
                                            adv_net, None, None, random_layer, domain_input)
        else:
            raise ValueError('Method cannot be recognized.')

        # total loss and backpropagation
        loss = config['lambda_adv'] * trans_loss + mlp_loss +\
               config['lambda_node'] * gnn_loss + config['lambda_edge'] * edge_loss
        loss.backward()
        optimizer.step()
        # printout train loss
        if i % 20 == 0 or i == config['adapt_iters'] - 1:
            log_str = 'Iters:(%4d/%d)\tMLP loss: %.4f\t GNN Loss: %.4f\t Edge Loss: %.4f\t Transfer loss:%.4f' % (
                i, config["adapt_iters"], mlp_loss.item(), config['lambda_node'] * gnn_loss.item(),
                config['lambda_edge'] * edge_loss.item(), config['lambda_adv'] * trans_loss.item()
            )
            utils.write_logs(config, log_str)
        # evaluate network every test_interval
        if i % config['test_interval'] == config['test_interval'] - 1:
            evaluate(i, config, base_network, classifier_gnn, dset_loaders['target_test'])

    return base_network, classifier_gnn


def upgrade_source_domain(config, max_inherit_domain, dsets, dset_loaders, base_network, classifier_gnn):
    target_dataset = ImageList(image_root=config['data_root'], image_list_root=config['data']['image_list_root'],
                               dataset=max_inherit_domain, transform=config['prep']['test'], domain_label=0,
                               dataset_name=config['dataset'], split='train')
    target_loader = DataLoader(target_dataset, batch_size=config['data']['test']['batch_size'],
                               num_workers=config['num_workers'], drop_last=False)
    # set networks to eval mode
    base_network.eval()
    classifier_gnn.eval()
    test_res = eval_domain(config, target_loader, base_network, classifier_gnn)

    # print out logs for domain
    log_str = 'Adding pseudo labels of dataset: %s\tPseudo-label acc: %.4f (%d/%d)\t Total samples: %d' \
              % (max_inherit_domain, test_res['pseudo_label_acc'] * 100., test_res['correct_pseudo_labels'],
                 test_res['total_pseudo_labels'], len(target_loader.dataset))
    config["out_file"].write(str(log_str) + '\n\n')
    config["out_file"].flush()
    print(log_str + '\n')

    # sub sample the dataset with the chosen confident pseudo labels
    pseudo_source_dataset = ImageList(image_root=config['data_root'],
                                      image_list_root=config['data']['image_list_root'],
                                      dataset=max_inherit_domain, transform=config['prep']['source'],
                                      domain_label=0, dataset_name=config['dataset'], split='train',
                                      sample_masks=test_res['sample_masks'], pseudo_labels=test_res['pred_cls'])

    # append to the existing source list
    dsets['source'] = ConcatDataset((dsets['source'], pseudo_source_dataset))
    # create new source dataloader
    dset_loaders['source'] = DataLoader(dsets['source'], batch_size=config['data']['source']['batch_size'] * 2,
                                        shuffle=True, num_workers=config['num_workers'],
                                        drop_last=True, pin_memory=True)

