from torch.utils.data import Dataset, DataLoader
from Utils import set_seed

g = set_seed()


# The data and labels were already loaded and stored in objects. This dataloader accesses a given object to return a
# sample and its label given an index.
class DataLoaderLabels(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label

    def __len__(self):
        return len(self.labels)


# This dataloader returns a sample without its label. This was used for loading batches of samples from val and test
# sets *without* shuffling (see init_data_loaders_no_labs_va_te function below).
class DataLoaderNoLabels(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

    def __len__(self):
        return self.data.size(0)


# This function initialises data loaders for training, validation, and testing. It follows the constraint that val and
# test samples are loaded without shuffling and their labels are not returned.
def init_data_loaders_no_labs_va_te(data_loader_class_tr, data_loader_class_eval, data_tr, labs_tr, data_va, data_te,
                                    batch_size):

    loader_tr = DataLoader(data_loader_class_tr(data_tr, labs_tr), batch_size=batch_size, shuffle=True,
                           num_workers=0, generator=g)
    loader_va = DataLoader(data_loader_class_eval(data_va), batch_size=batch_size, shuffle=False,
                           num_workers=0, generator=g)
    loader_te = DataLoader(data_loader_class_eval(data_te), batch_size=batch_size, shuffle=False,
                           num_workers=0, generator=g)

    return loader_tr, loader_va, loader_te
