from pathlib import Path
import shutil
import torch

def setup(args):
    job_dir = Path(args.output_root).joinpath(args.job_name)
    args.log_file = job_dir.joinpath("logs.txt")

    assert args.num_layers == len(args.train_fanouts)
    assert args.num_layers == len(args.batchwise_test_fanouts)

    do_ddp = args.total_num_nodes > 1 or args.one_node_ddp
    args.do_ddp = do_ddp
    print('do_ddp ==', do_ddp)
    if not do_ddp:
        if job_dir.exists():
            assert job_dir.is_dir()
            if args.overwrite_job_dir:
                shutil.rmtree(job_dir)
            else:
                raise ValueError(
                    f'job_dir {job_dir} exists. Use a different job name ' +
                    'or set --overwrite_job_dir')
        job_dir.mkdir(parents=True)

    num_devices_per_node = min(args.max_num_devices_per_node,
                               torch.cuda.device_count())
    args.num_devices_per_node = num_devices_per_node

    print(f'Using {num_devices_per_node} devices per node')

    if args.train_sampler == 'NeighborSampler' and args.train_type != 'serial':
        raise ValueError(
            'The dp (data parallel) train_type is not supported by this ' +
            'driver for the train_sampler NeighborSampler.')
    
    return args