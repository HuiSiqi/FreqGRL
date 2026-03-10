import os
import argparse
import json
from copy import deepcopy
import random

tar_set='cub'
number=5
split='val'

def parse_args():
    parser = argparse.ArgumentParser(description= 'processing json files')
    parser.add_argument('--file', default=f'/data4/pikey/code/MetaChannelMasking/filelists/{tar_set}/{split}.json')
    parser.add_argument('--outfile', default=f'/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_{split}_{tar_set}_{number}_aux.json')
    parser.add_argument('--number', type=int, default=number, help='image number for each class')
    parser.add_argument('--seed', type=int, default=42, help='random seed for sampling')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # NEW: 设定随机种子，保证每次运行抽样一致
    random.seed(args.seed)

    if args.file.endswith('.json'):
        # Create the file path
        file_path = args.file

        # Read the JSON file
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        files = deepcopy(json_data['image_names'])
        labels = deepcopy(json_data['image_labels'])

        bucket = {}
        if tar_set != 'cars' and tar_set != 'omniglot':
            for f in files:
                dirname = os.path.dirname(f)
                if dirname not in bucket:
                    bucket[dirname] = [f]
                else:
                    bucket[dirname].append(f)
        else:
            for f,label in zip(files,labels):
                if label not in bucket:
                    bucket[label] = [f]
                else:
                    bucket[label].append(f)

        out = {'image_names':[], 'image_labels':[]}
        for i,(_, files) in enumerate(bucket.items()):
            out['image_names'] += random.sample(files,args.number)
            out['image_labels'] += [i]*args.number

    with open(args.outfile, 'w') as f:
        json.dump(out,f)


