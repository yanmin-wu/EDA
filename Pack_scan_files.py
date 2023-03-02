import argparse
from src.joint_det_dataset import save_data

parser = argparse.ArgumentParser()
parser.add_argument('--scannet_data', required=True)
parser.add_argument('--data_root', required=True)
args, _ = parser.parse_known_args()

split = ['val', 'train']

for sp in split:
    print('Start packing the ' + sp + ' set...')
    save_data(f'{args.data_root}/{sp}_v3scans.pkl', sp, args.scannet_data)
    print('The ' + sp + ' set is packed!')
