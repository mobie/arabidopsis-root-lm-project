import json
import os
from concurrent import futures

import numpy as np
from elf.io import open_file
from pybdv.util import get_key

PATH = '/g/kreshuk/wolny/Datasets/LRP_Mamut/2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous_fused_cropped_export.h5'


def find_contrast_limits(setup_id, upper_percentile=99):
    if os.path.exists('./contrast_limits.json'):
        with open('./contrast_limits.json') as f:
            clims = json.load(f)
        clims = {int(k): v for k, v in clims.items()}
        if setup_id in clims:
            return clims[setup_id]
    else:
        clims = {}

    n_timepoints = 52

    with open_file(PATH, 'r') as f:

        def _process_tp(tp):
            print("Processing", tp)
            key = get_key(True, timepoint=tp, setup_id=setup_id, scale=0)
            vol = f[key][:]
            upper = np.percentile(vol, upper_percentile)
            return upper

        with futures.ThreadPoolExecutor(16) as tp:
            upper_limits = list(tp.map(_process_tp, range(n_timepoints)))

        upper_lim = np.mean(upper_limits)

    clims[setup_id] = [0., upper_lim]
    with open('./contrast_limits.json', 'w') as f:
        json.dump(clims, f)

    return clims[setup_id]


if __name__ == '__main__':
    print(find_contrast_limits(0))
    print(find_contrast_limits(1))
