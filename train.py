import os
import time
import torch
import random
import numpy as np
import torch.optim
from copy import deepcopy
from methods import backbone
from utils import green_text
from methods.gnnnet import GnnNet
from data.finetune_manager import FinetuneLoader
from data.datamgr import SimpleDataManager, SetDataManager
import importlib

from options import parse_args, get_resume_file, load_warmup_state

def train(S_base_loader, A_base_loader, A_val_loader, model, start_epoch, stop_epoch, params):
  param = model.split_model_parameters()
  # get optimizer and checkpoint path
  optimizer = torch.optim.Adam(param)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # for validation
  max_acc = 0
  total_it = 0
  start_time = time.time()
  spent = 0
  file = os.path.join(model.tf_path, 'val_acc.txt')
  max_tar_acc = 0
  for epoch in range(start_epoch,stop_epoch):
    model.train()
    total_it = model.train_loop(epoch, S_base_loader, A_base_loader, optimizer, total_it)
    model.eval()

    with torch.no_grad():
      tar_val_acc, tar_val_interval, tar_mask_val_acc, tar_mask_val_interval = model.test_loop(A_val_loader, prefix='Target Val')
    if tar_val_acc > max_tar_acc:
      print("best model! save...")
      max_tar_acc = tar_val_acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
    else:
      print("GG! best Validation accuracy {:f}".format(max_tar_acc))

    if ((epoch + 1) % params.save_freq==0) or (epoch==stop_epoch-1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
      torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    with open(file, 'a') as f:
      f.write(
        f'epoch {epoch} tar_val_acc {tar_val_acc:.3f} inter {tar_val_interval:.3f} tar_val_mask_acc {tar_mask_val_acc:.3f} inter {tar_mask_val_interval:.3f}\n')

    # count time
    end_time = time.time()
    cost = end_time-start_time-spent
    spent = spent + cost
    avg_cost = spent / (epoch + 1)
    print(green_text(
      f'Epoch:{epoch}') + f' | spent:{spent / 7200:.2f}h  rest:{avg_cost * (stop_epoch - epoch) / 7200:.2f}h cost:{cost:.2f}s')
  return model

# --- main function ---
if __name__=='__main__':
  # set numpy random seed
  seed = 0
  print("set seed = %d" % seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

  # parser argument
  params = parse_args('train')
  print('--- baseline training: {} ---\n'.format(params.name))
  print(params)

  # output and tensorboard dir
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  params.tf_dir = os.path.join(params.checkpoint_dir,'log')
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  loaders = FinetuneLoader(params)

  assert(params.modelType=='Student')
  print('meta-training the student model ME-D2N.')

  # source episode
  print('base source dataset: miniImagenet')
  S_base_loader  = loaders.S_Base_FS

  # target episode
  print('auxiliary target dataset: {} with num_target as {}', format(params.target_set, str(params.target_num_label)))
  A_base_loader  = loaders.A_Base_FS

  A_val_loader = loaders.A_Val_Full_FS

  assert(params.modelType=='Student')
  print('meta-training the student model ME-D2N.')

  # expert models
  print('--loading teacher models--')
  #define experts teacher model
  train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
  test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

  # student model
  assert(params.modelType=='Student')
  print('--meta-training the student model ME-D2N--')
  if params.method != None:
    GnnNetStudent = importlib.import_module(f'methods.{params.method}').GnnNetStudent
  else:
    GnnNetStudent = importlib.import_module(f'methods.PrototypeMethod').GnnNetStudent

  if params.backbone != None:
    model_dict = importlib.import_module(f'methods.{params.backbone}').model_dict
  else:
    model_dict = importlib.import_module(f'methods.backbone').model_dict

  #define student model
  model = GnnNetStudent( model_dict[params.model],params, tf_path=params.tf_dir, target_set = params.target_set, **train_few_shot_params)
  model = model.cuda()
  print(model)
  model.train()
  # load student model
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      tmp = torch.load(resume_file)
      start_epoch = tmp['epoch']+1
      model.load_state_dict(tmp['state'],strict=False)
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
  if params.warmup == 'gg3b0':
    raise Exception('Must provide the pre-trained feature encoder file using --warmup option!')
  state = load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup))
  model.feature.load_state_dict(state, strict=False)
  # training
  print('\n--- start the training ---')
  model = train(S_base_loader, A_base_loader, A_val_loader, model, start_epoch, stop_epoch, params)