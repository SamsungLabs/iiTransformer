# Code adapted from KAIR (https://github.com/cszn/KAIR)
# select dataset


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['dncnn', 'denoising']:
        from data.dataset_dncnn import DatasetDnCNN as D
    elif dataset_type in ['sr', 'super-resolution']:
        from data.dataset_sr import DatasetSR as D
    elif dataset_type in ['plain']:
        from data.dataset_plain import DatasetPlain as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
