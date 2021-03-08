import argparse
import os
from concurrent import futures
from subprocess import run

import yaml
import numpy as np
import vigra
from elf.io import open_file
from pybdv.util import get_key

PYTHON = '/g/kreshuk/pape/Work/software/conda/miniconda3/envs/plant-seg/bin/python'
PLANTSEG = '/g/kreshuk/pape/Work/my_projects/plant-seg/plantseg/run_plantseg.py'

PATH = os.path.join('/g/kreshuk/wolny/Datasets/LRP_Mamut',
                    '2017-08-02_17.49.34_stPVB003-2-2xDR5v2_F3_nb25_Marvelous_fused_cropped_export.h5')
TEMPLATE_CONFIG = os.path.join(os.path.abspath(os.path.split(__file__)[0]),
                               'plantseg_config_lrp.yaml')


class ChangeDir:
    def __init__(self, directory):
        self.directory = directory
        self.pwd = None

    def __enter__(self):
        self.pwd = os.getcwd()
        os.chdir(self.directory)

    def __exit__(self, type, value, traceback):
        os.chdir(self.pwd)


def segment_timepoint(timepoint, gpu=None):
    # create the input data for this timepoint
    tmp_folder = 'tmp_plantseg/tp_%03i' % timepoint
    os.makedirs(tmp_folder, exist_ok=True)

    if gpu is not None:
        assert isinstance(gpu, int)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    with ChangeDir(tmp_folder):
        raw_path = './raw.h5'
        config_path = './config.yaml'

        res_path = './segmentation.h5'
        if os.path.exists(res_path):
            return

        with open(TEMPLATE_CONFIG, 'r') as f:
            config = yaml.load(f)
        config['path'] = raw_path
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        with open_file(raw_path, 'a') as f:
            if 'raw' not in f:
                key = get_key(is_h5=True, timepoint=timepoint, setup_id=0, scale=0)
                with open_file(PATH, 'r') as f_in:
                    raw = f_in[key][:]
                f.create_dataset('raw', data=raw, chunks=(32, 128, 128))

        cmd = [PYTHON, PLANTSEG, '--config', config_path]
        run(cmd)

        print("Run post-processing ...")
        seg_path = 'PreProcessing/generic_light_sheet_3d_unet/MultiCut/raw_predictions_multicut.h5'
        with open_file(seg_path, 'r') as f, open_file(res_path, 'a') as f_out:
            seg = f['segmentation'][:]
            ids, sizes = np.unique(seg, return_counts=True)
            if 0 in ids:
                ids += 1
                seg += 1
            bg_id = ids[np.argmax(sizes)]
            seg[seg == bg_id] = 0
            seg = seg.astype('uint32')
            vigra.analysis.relabelConsecutive(seg, out=seg, start_label=1, keep_zeros=True)
            f_out.create_dataset('data', data=seg, compression='gzip')


def generate_segmentations(gpus):
    from create_project import timepoints_and_channels
    nt, _ = timepoints_and_channels(PATH)

    with futures.ProcessPoolExecutor(len(gpus)) as pp:
        gpu_id = 0
        tasks = []
        for tp in range(nt):
            gpu = gpus[gpu_id]
            tasks.append(pp.submit(segment_timepoint, tp, gpu))
            gpu_id += 1
            if gpu_id % len(gpus) == 0:
                gpu_id = 0
        [t.result() for t in tasks]


def check_result(timepoint):
    import napari

    print("Checking results for timepoint", timepoint)

    halo = [64, 384, 384]

    key = get_key(is_h5=True, timepoint=timepoint, setup_id=0, scale=0)
    with open_file(PATH, 'r') as f:
        ds = f[key]
        bb = tuple(slice(sh // 2 - ha, sh // 2 + ha) for sh, ha in zip(ds.shape, halo))
        raw = ds[bb]

    pred_path = os.path.join('tmp_plantseg/tp_%03i/PreProcessing' % timepoint,
                             'generic_light_sheet_3d_unet/PostProcessing/raw_predictions.h5')
    with open_file(pred_path, 'r') as f:
        pred = f['predictions'][bb]

    seg_path = 'tmp_plantseg/tp_%03i/segmentation.h5' % timepoint
    with open_file(seg_path, 'r') as f:
        seg = f['data'][bb]

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        viewer.add_image(pred)
        viewer.add_labels(seg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timepoint', '-t', default=None, type=int)
    parser.add_argument('--gpus', '-g', nargs='+', type=int, default=list(range(8)))
    parser.add_argument('--view', '-v', default=0, type=int)

    args = parser.parse_args()

    if args.timepoint is None:
        generate_segmentations(args.gpus)
    else:
        tp = args.timepoint
        if bool(args.view):
            check_result(tp)
        else:
            segment_timepoint(tp)
