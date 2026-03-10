import json
import os
import argparse
from copy import deepcopy
def parse_args():
  parser = argparse.ArgumentParser(description= 'processing json files')
  parser.add_argument('--file', default='/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/labled_base_cars_5.json')
  parser.add_argument('--outfile', default='/data/pikey/code/FSL/CDFSL/ME-D2N_for_CDFSL-main/output/pikey_labled_base_cars_5.json')
  parser.add_argument('--before', default='/DATACENTER/4/lovelyqian/CROSS-DOMAIN-FSL-DATASETS/cars/source/cars_train',help='directory of previous dataset')
  parser.add_argument('--after', default='/data/pikey/dataset/FewshotData/cars/Stanford_Cars/cars_train',help='directory of current dataset')
  return parser.parse_args()
# Iterate over the files in the directory

if __name__ == '__main__':
    args = parse_args()
    if args.file.endswith('.json'):
        # Create the file path
        file_path = args.file

        # Read the JSON file
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        out = deepcopy(json_data)
        new_paths = []
        for i in out['image_names']:
            new_paths.append(i.replace(args.before,args.after))
        out['image_names'] = new_paths

    with open(args.outfile, 'w') as f:
        json.dump(out,f)