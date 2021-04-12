from mobie.migration.migrate_v2 import migrate_project


def parse_menu_name(source_type, source_name):
    if source_type == 'image':
        return 'lm'
    else:
        return 'lm-segmentation'


migrate_project('./data', parse_menu_name)
