import torch
import os
import h5py

from methods import backbone

from data.datamgr import SimpleDataManager
from options import parse_args, get_best_file, get_assigned_file
import data.feature_loader as feat_loader
import random
import numpy as np
import importlib

# extract and save image features
def save_features(model, data_loader, featurefile, featurefile2):
    f = h5py.File(featurefile, 'w')
    f2 = h5py.File(featurefile2, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_labels2 = f2.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    all_feats2 = None
    count = 0
    count2 = 0

    for i, (x, y) in enumerate(data_loader):
        if (i % 10) == 0:
            print('    {:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        feats = model.standard_path(x,domain=model.args.ffilter)
        try:
            feats2 = model.masked_path(x)
        except:
            feats2 = feats

        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

        if all_feats2 is None:
            all_feats2 = f2.create_dataset('all_feats', [max_count] + list(feats2.size()[1:]), dtype='f')
        all_feats2[count2:count2 + feats2.size(0)] = feats2.data.cpu().numpy()
        all_labels2[count2:count2 + feats2.size(0)] = y.cpu().numpy()
        count2 = count2 + feats2.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count
    f.close()

    count_var2 = f2.create_dataset('count', (1,), dtype='i')
    count_var2[0] = count2
    f2.close()

# evaluate using features
def feature_evaluation(cl_data_file, cl_data_file2, model, n_way=5, n_support=5, n_query=15):
    def get_acc(scores, gt):
        pred = scores.data.cpu().numpy().argmax(axis=1)
        acc = np.mean(pred == gt) * 100
        return acc

    class_list = cl_data_file.keys()
    select_class = random.sample(class_list, n_way)
    z_all = []
    z_all2 = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        img_feat2 = cl_data_file2[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])
        z_all2.append([np.squeeze(img_feat2[perm_ids[i]]) for i in range(n_support + n_query)])

    z_all = torch.from_numpy(np.array(z_all))
    z_all2 = torch.from_numpy(np.array(z_all2))
    model.n_query = n_query
    scores_unmask = model.feat_predict(z_all)
    scores_mask = model.feat_predict(z_all2)
    scores_mix = (scores_mask + scores_unmask) / 2.0
    y = np.repeat(range(n_way), n_query)
    acc_mix = get_acc(scores_mix, y)
    acc_mask = get_acc(scores_mask, y)
    acc_unmask = get_acc(scores_unmask, y)
    return acc_mix, acc_mask, acc_unmask


def get_statics(acc):
    acc = np.asarray(acc)
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)
    acc_interval = 1.96 * acc_std / np.sqrt(len(acc))
    return acc_mean, acc_interval

# --- main ---
if __name__ == '__main__':
    # seed
    seed = 0
    print("set seed = %d" % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # parse argument
    params = parse_args('test')
    print(
        'Testing! {} shots on {} dataset with {} epochs of {}'.format(params.n_shot, params.dataset, params.save_epoch,
                                                                      params.name))
    remove_featurefile = True

    print('\nStage 1: saving features')
    # dataset
    print('  build dataset')
    image_size = 224
    split = params.split
    loadfile = os.path.join(params.data_dir, params.dataset, split + '.json')
    print('load file:', loadfile)
    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    print('  build feature encoder')
    checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
    if params.save_epoch != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
    else:
        modelfile = get_best_file(checkpoint_dir)

    # feature encoder
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    # student model
    print('--meta-training the student model ME-D2N--')
    if params.method != None:
        GnnNet = importlib.import_module(f'methods.{params.method}').GnnNetStudent
    else:
        GnnNet = importlib.import_module(f'methods.PrototypeMethod').GnnNetStudent

    if params.backbone != None:
        model_dict = importlib.import_module(f'methods.{params.backbone}').model_dict
    else:
        model_dict = importlib.import_module(f'methods.backbone').model_dict

    model = GnnNet(model_dict[params.model], params, **few_shot_params, target_set=params.dataset)
    model = model.cuda()
    tmp = torch.load(modelfile,map_location='cuda:0')
    state = tmp['state']
    print(state.keys())
    model.load_state_dict(state,strict=False)
    model.eval()

    # save feature file
    print('  extract and save features...')
    if params.save_epoch != -1:
        featurefile = os.path.join(checkpoint_dir, split + "_" + str(params.save_epoch) + ".hdf5")
        featurefile2 = os.path.join(checkpoint_dir, split + "_" + str(params.save_epoch) + "2.hdf5")
    else:
        featurefile = os.path.join(checkpoint_dir, split + ".hdf5")
        featurefile2 = os.path.join(checkpoint_dir, split + "2.hdf5")
    dirname = os.path.dirname(featurefile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    # if not os.path.exists(featurefile):
    save_features(model, data_loader, featurefile, featurefile2)

    print('\nStage 2: evaluate')
    acc_all = []
    iter_num = 1000
    # load feature file
    print('  load saved feature file')
    cl_data_file = feat_loader.init_loader(featurefile)
    cl_data_file2 = feat_loader.init_loader(featurefile2)

    # start evaluate
    print('  evaluate')
    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, cl_data_file2, model, n_query=15, **few_shot_params)
        acc_all.append(acc)
    acc_all = np.array(acc_all)
    # statics
    print('  get statics')
    statics = [get_statics(acc_all[:, i]) for i in range(acc_all.shape[1])]

    log = f'  {iter_num} test iterations: mixAcc = {statics[0][0]:.2f}% +- {statics[0][1]:.2f}  maskAcc = {statics[1][0]:.2f}% +- {statics[1][1]:.2f}  unmaskAcc = {statics[2][0]:.2f}% +- {statics[2][1]:.2f}'
    print(log)

    with open(os.path.join(checkpoint_dir,'test.txt'), 'w') as f:
        f.write(log)
    # remove feature files [optional]
    if remove_featurefile:
        os.remove(featurefile)