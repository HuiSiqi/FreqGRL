import numpy as np
import os
import glob
import torch
import argparse

def parse_args(script):
  parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
  parser.add_argument('--dataset', default='miniImagenet', help='miniImagenet/cub/cars/places/plantae')
  parser.add_argument('--image_size', default=224, type=int)
  parser.add_argument('--model', default='ResNet10', help='model: Conv{4|6} / ResNet{10|18|34}') # we use ResNet10 in the paper
  parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
  parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
  parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
  parser.add_argument('--train_episode'      , default=100, type=int,  help='training episode')
  parser.add_argument('--test_episode'      , default=100, type=int,  help='training episode')
  parser.add_argument('--train_aug', default=False, action='store_true',  help='perform data augmentation or not during training ')
  parser.add_argument('--name' , default='tmp', type=str, help='')
  parser.add_argument('--save_dir'    , default='./output', type=str, help='')
  parser.add_argument('--data_dir'    , default='filelists/', type=str, help='')   #TO change your dataset here
  parser.add_argument('--alpha'      , default=0.1, type=float,  help='lower band reject filter threshold')
  parser.add_argument('--beta'      , default=1.0, type=float,  help='upper band reject filter threshold')
  parser.add_argument("--drop_layers", nargs='+', default=[], type=int, help="meta layers")
  parser.add_argument("--drop_bn_layers", nargs='+', default=[], type=int, help="drop bn layers")
  parser.add_argument('--drop_type', default='static', type=str, help='channel mask type should be in ["static","MSD","MGD"]')
  parser.add_argument('--aug_rate', default=0.5, type=float, help='dssc augment channel percentage')
  parser.add_argument('--drop_rate', default=0.75, type=float, help='dropout ratio')
  parser.add_argument('--drop_prob', default=1.0, type=float, help='probability of aplly dropout')
  parser.add_argument('--fixteacher', action='store_true', help='use static teacher to train the student')
  parser.add_argument('--param', default=0.3, type=float, help='initial param of MGD')
  parser.add_argument('--prefix', default='baseline', type=str, help='initial param of MGD')
  parser.add_argument('--outdim', default=512, type=int, help='initial param of MGD')
  parser.add_argument('--method', default=None, type=str, help='which method')
  parser.add_argument('--backbone', default=None, type=str, help='which backbone')
  parser.add_argument('--save_name', default='', type=str, help='save file name')
  parser.add_argument('--grad_file', default='', type=str, help='file to load channel gradients')
  parser.add_argument('--se_layers', nargs='+', default=[], type=int, help='residual blocks that use se_layer')
  parser.add_argument('--FM', action='store_true', help='use frequency mixing')
  parser.add_argument('--gamma', default=0.2, type=float, help='low frequency thresholds')
  parser.add_argument('--FE', action='store_true', help='use frequency enhancing')
  parser.add_argument('--FF', action='store_true', help='use frequency filtering')
  parser.add_argument('--FE_low', default=0.5,type=float, help='low frequency threshold')
  parser.add_argument('--FE_high', default=1.0,type=float, help='high frequency threshold')
  parser.add_argument('--ablate_highfreqconv', default=None, type=str, help='structure of high-freq-conv')

  if script == 'train':
    parser.add_argument('--num_classes' , default=64, type=int, help='total number of classes in softmax, only used in baseline')
    parser.add_argument('--save_freq'   , default=200, type=int, help='Save frequency')
    parser.add_argument('--target_set', default='cub', help='cub/cars/places/plantae, use the extremely labeled target data')
    parser.add_argument('--modelType', default='St-Net', help='pretrain/St-Net/Tt-Net/Student')
    parser.add_argument('--ckp_S'      , default='', type=str,help='the ckp path of the expert St-Net model')
    parser.add_argument('--ckp_A'      , default='', type=str,help='the ckp path of the expert Tt-Net model')
    parser.add_argument('--target_num_label', default=5, type=int, help='number of labeled target base images per class')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch')
    parser.add_argument('--resume'      , default='', type=str, help='continue from previous trained model with largest epoch')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='')
    parser.add_argument('--warmup'      , default='gg3b0', type=str, help='continue from baseline, neglected if resume is true')
    parser.add_argument('--temp'      , default=1, type=float, help='temperature of prototypical loss in contrastive space')
    parser.add_argument('--meta' , default=False, action='store_true', help='wheather drop generalizable')
    parser.add_argument('--wdif' , default=0.0, type=float, help='different loss weight')
    parser.add_argument('--lamb' , default=0.0, type=float, help='weight of the source data in domain mixup')
    parser.add_argument('--wsim' , default=0.0, type=float, help='different loss weight')
    parser.add_argument('--sigma' , default=0.2, type=float, help='different loss weight')
    parser.add_argument('--worth' , default=0.1, type=float, help='orthogonal loss weight')
    parser.add_argument('--aug_type' , default='cutmix', type=str, help='image augmentation type')
    parser.add_argument('--reverse', default=False, action='store_true', help='wheather augment negligible channel')
    parser.add_argument('--taylor_expansion', default='init', help='point where apply taylor expansion of target few shot loss of channel mask, choose from [init, ones, zeros, random]')
    parser.add_argument('--fix', default=False, help='wheather search start from fix point')
    parser.add_argument('--w_fisher' , default=0.001, type=float, help='fisher penulty weight')
    parser.add_argument('--momentum' , default=0.9, type=float, help='fisher update momentum')
    parser.add_argument('--ffilt_S' , action='store_true', help='filt source frequency')
    parser.add_argument('--ffilt_A' , action='store_true', help='filt target frequency')
    parser.add_argument('--A_ffilt_S' , action='store_true', help='filt target frequency')
    parser.add_argument('--S_ffilt_A' , action='store_true', help='filt target frequency')
    parser.add_argument('--share_ffilt' , action='store_true', help='filt target frequency')
    parser.add_argument('--ablate_exp' , default='src_low_tar_high', type=str, help='ablate experiment name')


  elif script == 'test':
    parser.add_argument('--target_set', default='cub', help='cub/cars/places/plantae, use the extremely labeled target data')
    parser.add_argument('--target_training', action='store_true', help='filt target frequency')
    parser.add_argument('--split', default='novel', help='base/val/novel')
    parser.add_argument('--save_epoch', default=400, type=int,help ='load the model trained in x epoch, use the best model if x is -1')
    parser.add_argument('--warmup'      , default='gg3bo', type = str, help = 'just for insert the test function into the training.')
    parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch')
    parser.add_argument('--amp_mode'  , default='full', type=str, help ='choose from ["full", "high", "low"]')
    parser.add_argument('--phase_mode'  , default='full', type=str, help ='choose from ["full", "high", "low"]')
    parser.add_argument('--r'  , default=0.2, type=float, help ='low frequency ratio')
    parser.add_argument('--ffilter' , default='A', type=str, help='which filter domain')
  else:
    raise ValueError('Unknown script')

  return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
  assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
  return assign_file

def get_resume_file(checkpoint_dir, resume_epoch=-1):
  filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
  print(filelist)
  if len(filelist) == 0:
    return None

  filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
  epochs = []
  for x in filelist:
    try: e = int(os.path.splitext(os.path.basename(x))[0])
    except: e = 0
    epochs.append(e)
  epochs = np.array(epochs)
  max_epoch = np.max(epochs)
  epoch = max_epoch if resume_epoch == -1 else resume_epoch
  resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
  return resume_file

def get_best_file(checkpoint_dir):
  best_file = os.path.join(checkpoint_dir, 'best_model.tar')
  if os.path.isfile(best_file):
    return best_file
  else:
    return get_resume_file(checkpoint_dir)

def load_warmup_state(filename):
  print('  load pre-trained model file: {}'.format(filename))
  warmup_resume_file = get_resume_file(filename)
  print(' warmup_resume_file:', warmup_resume_file)
  tmp = torch.load(warmup_resume_file,map_location='cuda:0')
  if tmp is not None:
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
      if 'feature.' in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      else:
        state.pop(key)
  else:
    raise ValueError(' No pre-trained encoder file found!')
  return state

def get_feature_state(checkpoint):
  tmp = torch.load(checkpoint)
  if tmp is not None:
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
      if 'feature.' in key:
        newkey = key.replace("feature.", "")
        state[newkey] = state.pop(key)
      else:
        state.pop(key)
  else:
    raise ValueError(' No pre-trained encoder file found!')
  return state

'''
def load_warmup_state(filename, method):
  print('  load pre-trained model file: {}'.format(filename))
  warmup_resume_file = get_resume_file(filename)
  print(' warmup_resume_file:', warmup_resume_file)
  tmp = torch.load(warmup_resume_file)
  if tmp is not None:
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
      if 'relationnet' in method and "feature." in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      elif method == 'gnnnet' and 'feature.' in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      else:
        state.pop(key)
  else:
    raise ValueError(' No pre-trained encoder file found!')
  return state
'''
