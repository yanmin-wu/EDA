"""
Customized script to download only the necessary ScanNet files.

You still need to get the original script from the authors of ScanNet.
"""

import os

from tqdm import tqdm


def get_scan_ids():
    """Load the .csv files and return a set of scan_ids."""
    scan_ids = []
    for name in ['train', 'val']:
        with open('data/meta_data/scannetv2_' + name + '.txt') as fid:
            scan_ids += fid.readlines()
    return sorted(list(set(sid.strip('\n') for sid in scan_ids)))


def download_scan_id(scan_id):
    """Download files for a specifed scan_id."""
    command = 'python download-scannet.py -o . --id %s' % scan_id
    to_download = [
        '.aggregation.json',
        '.txt',
        '_vh_clean_2.0.010000.segs.json',
        '_vh_clean_2.ply',
        '_vh_clean_2.labels.ply'
    ]
    for filetype in to_download:
        os.system(command + ' --type ' + filetype)


def main():
    """Download all necessary files for all scan_ids."""
    scan_ids = get_scan_ids()
    for scan_id in tqdm(scan_ids):
        download_scan_id(scan_id)


if __name__ == "__main__":
    main()
