from os.path import join
import datetime as dt

from dlib.utils.utils_parser import parse_input

from dlib.utils.tools import log_device
from dlib.utils.tools import bye

from dlib.utils import constants
from dlib.utils.shared import is_cc
import dlib.dllogger as DLLogger
from dlib.utils.utils_config import find_last_checkpoint
from dlib.utils.utils_config import save_config
from dlib.utils.utils_reproducibility import set_seed
from dlib.utils.utils_dataloaders import get_train_loader
from dlib.utils.utils_dataloaders import get_all_eval_loaders
from dlib.models.select_model import define_model
from dlib.utils.utils_tracker import find_last_tracker
from dlib.utils.utils_trainer import train_valid
from dlib.utils.utils_trainer import ddp_barrier


def main():
    args, args_dict = parse_input()
    log_device(args)

    init_iter_G, init_path_G = find_last_checkpoint(
        join(args.outd_backup, args.save_dir_models), net_type='G',
        pretrained_path=args.netG['init_pretrained_path'])
    args.netG['checkpoint_path_netG'] = init_path_G
    # todo: E.
    init_iter_optimizerG, init_path_optimizerG = find_last_checkpoint(
        join(args.outd_backup, args.save_dir_models), net_type='optimizerG')
    args.netG['checkpoint_path_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_optimizerG)

    set_seed(args.myseed, verbose=False)

    train_loader, train_sampler = get_train_loader(args, debug_n=-1)

    valid_loaders = get_all_eval_loaders(args, args.valid_dsets,
                                         n=args.valid_n_samples)

    n_test = -1
    test_loaders = get_all_eval_loaders(args, args.test_dsets, n=n_test)

    tracker, roi_tracker = find_last_tracker(args.outd_backup, args)

    model = define_model(args)
    model.init_train()

    if args.is_master:
        DLLogger.log(model.info_network())

    train_valid(args=args,
                model=model,
                train_loader=train_loader,
                train_sampler=train_sampler,
                valid_loaders=valid_loaders,
                test_loaders=test_loaders,
                tracker=tracker,
                roi_tracker=roi_tracker,
                current_step=current_step
                )

    if args.is_master:
        args.tend = dt.datetime.now()
        save_config(args, args.outd_backup)
        bye(args)

    ddp_barrier(args.distributed)


if __name__ == '__main__':
    main()
