import sys
from os.path import dirname, abspath
from typing import Union, Tuple

import torch.distributed as dist
import torch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


def sync_tensor_across_gpus(t: Union[torch.Tensor, None]
                            ) -> Union[torch.Tensor, None]:
    if t is None:
        return None
    group = dist.group.WORLD
    group_size = torch.distributed.get_world_size(group)
    gather_t_tensor = [torch.zeros_like(t) for _ in
                       range(group_size)]
    dist.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor, dim=0)


def sync_non_tensor_value_across_gpus(v):
    assert not torch.is_tensor(v)
    # todo: adjust dtype accordning to v type. for now: set to float.

    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    nxx = torch.tensor([v], dtype=torch.float,
                       requires_grad=False, device=device).view(1, )
    return sync_tensor_across_gpus(nxx).sum().item()


def sync_dict_across_gpus(holder: dict,
                          move_sync_vals_to_cpu: bool = False) -> dict:
    # holder: key: float(cpu). value: float(gpu).
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    keys_list = list(holder.keys())
    n = len(keys_list)
    assert all([isinstance(k, float) for k in keys_list])

    keys_tensor = torch.tensor(keys_list, dtype=torch.float,
                               requires_grad=False, device=device).view(n, )
    vals_list = [holder[k] for k in keys_list]
    assert all([k.device == device for k in vals_list])
    assert all([k.numel() == 1 for k in vals_list])  # single value.

    vals_tensor = torch.tensor(vals_list, dtype=vals_list[0].dtype,
                               requires_grad=False, device=device).view(n, )

    keys_sync = sync_tensor_across_gpus(keys_tensor)
    vals_sync = sync_tensor_across_gpus(vals_tensor)

    keys_sync_cpu = keys_sync.cpu()

    if move_sync_vals_to_cpu:
        vals_sync = vals_sync.cpu()

    out = dict()
    for i, k in enumerate(keys_sync_cpu):
        out[k.item()] = vals_sync[i]

    return out

