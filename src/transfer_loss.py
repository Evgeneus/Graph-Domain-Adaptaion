import numpy as np
import torch
import torch.nn as nn


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def Entropy(input):
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def CDAN(ndomains, input_list, ad_net, entropy=None, coeff=None, random_layer=None, dc_target=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        entropy_weight = []
        indices = []
        arange = torch.arange(entropy.shape[0])
        domain_weights = np.zeros(ndomains)
        for i in range(ndomains):
            domain_samples = entropy[dc_target == i]
            if domain_samples.shape[0] == 0:
                continue
            domain_index = arange[dc_target == i]
            entropy_weight.append(domain_samples / torch.sum(domain_samples).detach().item())
            indices.append(domain_index)
            domain_weights[i] = entropy.shape[0] / domain_samples.shape[0]
        indices = torch.argsort(torch.cat(indices))
        entropy_weight = torch.cat(entropy_weight, dim=0)
        entropy_weight_norm = entropy_weight[indices]
        ## to make the sum of weights to ndomains
        domain_weights = domain_weights * ndomains / np.sum(domain_weights)
        ad_loss = torch.sum(entropy_weight_norm.view(-1, 1) * nn.CrossEntropyLoss(weight=torch.from_numpy(domain_weights.astype(np.float32)).cuda(), reduction='sum')(ad_out, dc_target)) / (torch.sum(entropy_weight_norm).detach().item() * entropy.shape[0])
        return ad_loss
    else:
        return nn.CrossEntropyLoss()(ad_out, dc_target) 
