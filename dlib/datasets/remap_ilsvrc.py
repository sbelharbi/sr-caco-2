import sys
from os.path import join, dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


def load_mapping(path: str) -> dict:
    mapz = dict()
    with open(path, 'r') as fin:
        for line in fin.readlines():
            org_k, new_k = line.strip('\n').replace(' ', '').split(',')
            assert org_k not in mapz
            mapz[org_k] = new_k

    return mapz


if __name__ == '__main__':
    vlddir = join(root_dir, 'folds/wsol-done-right-splits/ILSVRC/val')

    maps = load_mapping(path=join(vlddir, 'mapping.txt'))

    for name in ['class_labels.txt', 'image_ids.txt', 'image_sizes.txt',
                 'localization.txt']:
        print(f'Updating {name}')
        path = join(vlddir, name)
        old_lines = []
        new_lines = []
        updated = False
        with open(path, 'r') as fin:
            for line in fin.readlines():
                old_lines.append(line)
                org_k = line.split(',')[0].replace('val2/', '').strip('\n')
                print(line)

                if org_k != maps[org_k]:
                    updated = True
                    line = line.replace(org_k, maps[org_k])
                    new_lines.append(line)
                    print(line)
                    print('************************')

        if updated:
            with open(path, 'w') as fin:
                for line in new_lines:
                    fin.write(line)



