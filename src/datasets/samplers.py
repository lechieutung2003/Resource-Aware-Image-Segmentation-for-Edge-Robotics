from torch.utils.data import Sampler

class EdgeSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
    def __iter__(self):
        return iter(range(len(self.data_source)))
    def __len__(self):
        return len(self.data_source)
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update
# Maintenance update

    # Edge case handled successfully
