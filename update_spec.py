import json
import os
from mobie.migration.migrate_v2 import migrate_project
from mobie.validation.utils import validate_with_schema


def parse_menu_name(source_type, source_name):
    if source_type == 'image':
        return 'lm'
    else:
        return 'lm-segmentation'


def update_sources(sources):
    new_sources = {}
    for name, source in sources.items():

        source_type = list(source.keys())[0]
        source = source[source_type]

        storage = source.pop('imageDataLocations')

        local = storage.pop('local')
        storage['fileSystem'] = local

        remote = storage.pop('remote', None)
        if remote is not None:
            storage['s3store'] = remote

        source['imageDataLocations'] = storage
        new_sources[name] = {source_type: source}
    return new_sources


def rename_image_data_properties():
    with open('./data/project.json') as f:
        dsets = json.load(f)['datasets']
        for dset in dsets:

            source_file = os.path.join('./data', dset, 'dataset.json')
            with open(source_file) as f:
                ds_meta = json.load(f)
            sources = update_sources(ds_meta['sources'])
            ds_meta['sources'] = sources
            validate_with_schema(ds_meta, 'dataset')

            with open(source_file, 'w') as f:
                json.dump(ds_meta, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    # TODO set the number of timepoints (52) in the dataset metadata
    # migrate_project('./data', parse_menu_name)

    # rename 'local' -> 'fileSystem', 'remote' -> 's3store'
    rename_image_data_properties()
