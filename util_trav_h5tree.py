"""Thoams Liao Oct, utility func to traverse h5py structure"""

import h5py
import numpy as np
# check structure of h5py file


def extract(a, b, show_value=False):
    print("---> name: ", a, "Group" if isinstance(b, h5py.Group) else "Dataset")
    print("    node: ", np.array(b) if show_value else b)

def traverse(file, show_value=False):
    with h5py.File(file, 'r') as hdf:
        hdf.visititems(extract)


if __name__ == "__main__":
    traverse('mini_test_embeddings.h5')