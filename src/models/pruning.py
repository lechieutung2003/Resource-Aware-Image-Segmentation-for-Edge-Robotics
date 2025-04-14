import torch.nn.utils.prune as prune

def prune_conv_layer(module, amount=0.3):
    prune.l1_unstructured(module, name='weight', amount=amount)
    prune.remove(module, 'weight')
# Maintenance update
# Maintenance update
# Maintenance update
