import os
import sys
from os.path import join, dirname, abspath
from tqdm import tqdm


from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils import constants

SPLIT = constants.VALIDSET


def get_ids(img_id_file: str) -> list:
    image_ids = []
    with open(img_id_file, 'r') as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n').replace('val2/', ''))
    return image_ids


def get_image_sizes(path_img_sz: str) -> dict:
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(path_img_sz, 'r') as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            image_id = image_id.replace('val2/', '')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


def compare_bforce_with_mapping(path_provided_map_1: str, bf: dict) -> list:
    mapz = dict()
    with open(path_provided_map_1, 'r') as fin:
        for line in fin.readlines():
            org_k, new_k = line.strip('\n').replace(' ', '').split(',')
            assert org_k not in mapz
            mapz[org_k] = new_k

    failed = []
    for k in bf:
        if bf[k] != mapz[k]:
            failed.append(f'{k}, {bf[k]}, {mapz[k]}')

    return failed


if __name__ == '__main__':
    # hard paths.
    vlddir = join(root_dir, 'folds/wsol-done-right-splits/ILSVRC/val')

    # original valid data.
    org_img_id_path = join(vlddir, 'image_ids.txt')
    org_img_sz_path = join(vlddir, 'image_sizes.txt')

    org_ids = get_ids(img_id_file=org_img_id_path)
    org_sz = get_image_sizes(path_img_sz=org_img_sz_path)

    # new valid data.
    data_valid = '/export/livia/home/vision/sbelharbi/transit/wsol-done-right' \
                 '/ILSVRC/val2'

    subfds = [x[0] for x in os.walk(data_valid) if x[0] != data_valid]
    subfds = [x.replace(data_valid + '/', '') for x in subfds]
    subfds.sort(key=int)
    new_ids = []
    new_sz = dict()
    mappings = dict()  # orig: new
    failed_mappings = []

    for fd in tqdm(subfds, ncols=80, total=len(subfds)):
        c_or_ids = [k for k in org_ids if k.startswith(fd + '/')]
        for file in os.listdir(join(data_valid, fd)):
            if file.endswith(".jpeg"):
                pfile = os.path.join(data_valid, fd, file)
                image = Image.open(pfile)
                w, h = image.size
                new_k = f'{fd}/{file}'
                new_ids.append(new_k)

                new_sz[new_k] = (w, h)

                # bf
                matchs = []
                for k in c_or_ids:
                    matchs.append(org_sz[k] == new_sz[new_k])

                if sum(matchs) == 1:
                    orig_k = c_or_ids[matchs.index(True)]
                    assert orig_k not in mappings
                    mappings[orig_k] = new_k
                else:
                    failed_mappings.append(new_k)

    with open('mapping.txt', 'w') as fout:
        for k in mappings:
            fout.write(f'{k}, {mappings[k]}\n')

    # compare bf results with the provided mapping.
    pathmp = join('/export/livia/home/vision/sbelharbi/transit/wsol-done'
                  '-right/mapping.txt')
    failed = compare_bforce_with_mapping(path_provided_map_1=pathmp,
                                         bf=mappings)

    print(f'BFORCE: found {len(list(mappings.keys()))} possibly correct pairs.')
    print(f'BFORCE: found {len(failed_mappings)} failed matching due to '
          f'duplicate '
          f'sizes.')

    print(f'found {len(failed)} failed comparison.')















