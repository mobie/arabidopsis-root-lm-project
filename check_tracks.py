import os
import numpy as np
from elf.io import open_file
from elf.tracking.mamut import extract_tracks_as_volume

RESOLUTION = [0.25, 0.1625, 0.1625]
ROOT = '/g/kreshuk/wolny/Datasets/LRP_Mamut'
TRACKS = os.path.join(ROOT,
                      '2018-05-28_Rmamut_manual_color_set_to_heatmap_DR5v2_from_2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous_fused_cropped_export-mamut.xml')

tp = 17
print("Load seg ...")
seg_path = './data/arabodopsis-root/images/local/lm-cells.n5'
with open_file(seg_path, 'r') as f:
    ds = f[f'setup0/timepoint{tp}/s0']
    ds.n_threads = 8
    shape = ds.shape

    print("Load tracks ...")
    tracks = extract_tracks_as_volume(TRACKS, tp, shape, RESOLUTION)
    track_ids, idx = np.unique(tracks, return_index=True)
    seg = ds[:]
    index = np.unravel_index(idx, seg.shape)
    seg_ids = seg[index]

print("Make track vol ...")
tracks_vol = np.zeros_like(seg)
for seg_id, track_id in zip(seg_ids, track_ids):
    tracks_vol[seg == seg_id] = track_id


import napari
with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_labels(seg)
    viewer.add_labels(tracks_vol)
