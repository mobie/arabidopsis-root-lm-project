import os
import numpy as np
import pandas as pd

from elf.io import open_file
from elf.tracking.mamut import extract_tracks_as_volume
from mobie import initialize_dataset, add_image_data, add_segmentation
from mobie.tables.default_table import compute_default_table
from pybdv.converter import convert_to_bdv
from pybdv.util import get_key

ROOT = '/g/kreshuk/wolny/Datasets/LRP_Mamut'
TRACKS = os.path.join(ROOT,
                      '2018-05-28_Rmamut_manual_color_set_to_heatmap_DR5v2_from_2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous_fused_cropped_export-mamut.xml')
DS_NAME = 'arabodopsis-root'

RESOLUTION = [0.25, 0.1625, 0.1625]
CHUNKS = (64, 64, 64)
SCALE_FACTORS = [
    [1, 2, 2],
    [1, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2]
]


def timepoints_and_channels(path):
    with open_file(path, 'r') as f:
        tps = list(f.keys())
        tps = [tp for tp in tps if tp.startswith('t0')]

        nc = None
        for tp in tps:
            g = f[tp]
            if nc is None:
                nc = len(g.keys())
            else:
                assert len(g.keys()) == nc

    return len(tps), nc


def add_timepoint(in_path, out_path, tp, channel, mode, in_key=None):
    if in_key is None:
        in_key = get_key(is_h5=True, timepoint=tp, setup_id=channel, scale=0)
    out_key = get_key(is_h5=False, timepoint=tp, setup_id=0, scale=0)

    # skip timepoints that have been copied already
    with open_file(out_path, 'r') as f:
        if out_key in f:
            return

    convert_to_bdv(in_path, in_key, out_path, SCALE_FACTORS,
                   downscale_mode=mode, resolution=RESOLUTION,
                   unit='micrometer', setup_id=0, timepoint=tp)


def add_lm_boundaries(path, n_timepoints):
    im_name = 'lm-membranes'
    out_path = f'./data/{DS_NAME}/images/local/{im_name}.n5'
    key0 = get_key(is_h5=True, timepoint=0, setup_id=0, scale=0)

    clim = [0., 3000.]  # TODO
    if not os.path.exists(out_path.replace('.n5', '.xml')):
        initialize_dataset(
            path, key0,
            './data', DS_NAME,
            raw_name=im_name,
            resolution=RESOLUTION,
            chunks=CHUNKS,
            scale_factors=SCALE_FACTORS,
            is_default=True,
            max_jobs=8,
            settings={'contrastLimits': clim}
        )

    assert os.path.exists(out_path)
    for tp in range(1, n_timepoints):
        add_timepoint(path, out_path, tp, channel=0,
                      mode='mean')


def add_lm_nuclei(path, n_timepoints):
    im_name = 'lm-nuclei'
    out_path = f'./data/{DS_NAME}/images/local/{im_name}.n5'
    key0 = get_key(is_h5=True, timepoint=0, setup_id=1, scale=0)

    clim = [0., 3000.]  # TODO
    if not os.path.exists(out_path.replace('.n5', '.xml')):
        add_image_data(
            path, key0,
            './data', DS_NAME,
            image_name=im_name,
            resolution=RESOLUTION,
            chunks=CHUNKS,
            scale_factors=SCALE_FACTORS,
            max_jobs=8,
            settings={'contrastLimits': clim}
        )

    assert os.path.exists(out_path)
    for tp in range(1, n_timepoints):
        add_timepoint(path, out_path, tp, channel=1,
                      mode='mean')


def extract_track_ids(timepoint, seg_path, key):
    with open_file(seg_path, 'r') as f:
        shape = f[key].shape
    tracks = extract_tracks_as_volume(TRACKS, timepoint, shape, RESOLUTION)

    with open_file(seg_path, 'r') as f:
        ds = f[key]
        ds.n_threads = 8
        seg = ds[:]
    ids = np.unique(seg)

    unique_track_ids, idx = np.unique(tracks, return_index=True)
    index = np.unravel_index(idx, shape)

    seg_ids = seg[index]
    track_ids = np.zeros(len(ids))
    track_ids[seg_ids] = unique_track_ids

    return track_ids


# update the table with a timepoint column
# and the track ids for this segmentation
def update_table(table, timepoint, seg_path, key):

    label_ids = table['label_id'].values
    tp_col = np.array([[timepoint]] * len(label_ids))
    track_ids = extract_track_ids(timepoint, seg_path, key)

    new_data = np.concatenate([label_ids, tp_col, track_ids], axis=1)
    new_columns = pd.DataFrame(data=new_data, columns=['label_id', 'timepoint', 'track_id'])
    table = table.merge(new_columns)
    return table


# compute the default table for each timepoint,
# add the track-id column and concatenate all to single table
def add_table(seg_paths, key, out_path):
    tmp_tables = './tmp_tables'

    table = None

    for tp, seg_path in enumerate(seg_paths):
        tmp_folder = os.path.join(tmp_tables, f'table{tp}')
        tmp_path = os.path.join(tmp_folder, 'table.csv')
        compute_default_table(seg_path, key, tmp_path,
                              resolution=RESOLUTION, tmp_folder=tmp_folder,
                              target='local', max_jobs=8)
        this_table = pd.read_csv(tmp_path, sep='\t')
        this_table = update_table(this_table, tp, seg_path, key)
        if table is None:
            table = this_table
        else:
            table = pd.concat([table, this_table])

    table.to_csv(out_path, sep='\t', index=False)


def add_segmentations(seg_paths):
    path0 = seg_paths[0]
    seg_name = 'lm-cells'
    out_path = f'./data/{DS_NAME}/images/local/{seg_name}.n5'
    key = 'data'

    if not os.path.exists(out_path.replace('.n5', '.xml')):
        add_segmentation(
            path0, key,
            './data', DS_NAME,
            segmentation_name=seg_name,
            resolution=RESOLUTION,
            chunks=CHUNKS,
            scale_factors=SCALE_FACTORS,
            max_jobs=8,
            add_default_table=False
        )

    assert os.path.exists(out_path)
    for tp, path in enumerate(seg_paths[1:], 1):
        add_timepoint(path, out_path, tp, channel=0, mode='nearest',
                      in_key=key)

    table_folder = f'./data/{DS_NAME}/tables/{seg_name}'
    os.makedirs(table_folder, exist_ok=True)
    table_path = os.path.join(table_folder, 'default.csv')
    add_table(seg_paths, key, table_path)


# TODO make bookmark(s) that include timepoint(s)?
def make_bookmarks():
    pass


def create_project():
    path = os.path.join(
        ROOT,
        '2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous_fused_cropped_export.h5'
    )

    nt, nc = timepoints_and_channels(path)

    # add_lm_boundaries(path, nt)
    # add_lm_nuclei(path, nt)

    seg_paths = [
        './tmp_plantseg/tp_%03i/segmentation.h5' % tp for tp in range(nt)
    ]
    add_segmentations(seg_paths)


if __name__ == '__main__':
    create_project()
    make_bookmarks()
