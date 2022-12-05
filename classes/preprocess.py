import argparse
import numpy as np
import os
import tqdm

from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path

class PreProcessor():

    def __init__(self, sample_dim, output_dir):
        self.sample_dim = sample_dim
        self.output_dir = output_dir
        

    def __call__(self, meta_line):
        name = meta_line[0]
        name = os.path.splitext(name)[0] # remove extension
        
        coordinates = meta_line[1:]
        for i in range(0, len(coordinates), 2):
            coordinates[i] /= self.sample_dim[0]
            coordinates[i+1] /= self.sample_dim[1]
        
        array_coord = np.array([coordinates])
        np.save(self.output_dir/f'{name}.npy', array_coord, allow_pickle=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--metadata', '-m', help='Metadata csv file location', default='metadata.csv')
    parser.add_argument('--output_dir', '-o', help='Outputs directory location', default='scaled_coordinates')
    args = parser.parse_args()

    metadata_file = open(Path(args.metadata))
    output_dir = Path(args.output_dir)
    lines = [line.rstrip() for line in metadata_file.readlines()]
    metadata_file.close()

    sample_dim = [int(element) for element in lines[0].split(",")]

    metadata = []
    for l in lines[1:]:
        line = l.split(",")
        name = line[0]
        for i in range(1,len(line)):
            line[i] = int(line[i])
        metadata.append(line)

    
    os.makedirs(output_dir, exist_ok=True)
    preproc = PreProcessor(sample_dim, output_dir)
    
    pool = Pool(processes=cpu_count())
    mapper = pool.imap_unordered(preproc, metadata)

    for i in tqdm.tqdm(iterable=mapper,desc='Preprocessing', total=len(metadata)):
        pass