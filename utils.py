import os
import json

import torch

def save_config(args):
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        args_dict = vars(args)
        if 'ds_config' in args:
            with open(args.ds_config, "r", encoding="utf-8") as ds_f:
                ds_config = json.load(ds_f)
                args_dict['ds_config'] = ds_config
        json.dump(args_dict, f)


def get_batch_config(args):
    with open(args.ds_config, "r", encoding="utf-8") as ds_f:
        ds_config = json.load(ds_f)
    args.per_device_batch_size = ds_config['per_device_batch_size']
    args.gradient_accumulation_steps = ds_config['gradient_accumulation_steps']

    return args

def set_value_to_shared_json_file(file_path, key, value, local_rank, writer_rank):
    """
        Blocks all the processors that are not writers & write key value to json file
        Very naive implementation of shared fuction!!
        Use this function at your own risk!!
    """
    # block all other ranks until writer finishs
    if local_rank != writer_rank:
        torch.distributed.barrier()
    else:
        if os.path.exists(os.path.join(file_path,'temp.json')):
            with open(os.path.join(file_path,'temp.json'), 'r') as f:
                temp_file = json.load(f)
                temp_file[key] = value
        else:
            temp_file = {}
        with open(os.path.join(file_path,'temp.json'), 'w') as f:
            json.dump({key: value}, f)
    # release
    if local_rank == writer_rank:
        torch.distributed.barrier()

def get_value_from_shared_json_file(file_path, key):
    """
        Very naive implementation of shared fuction!!
        Use this function at your own risk!!
    """
    try:
        with open(os.path.join(file_path,'temp.json'), 'r') as f:
            return json.load(f)[key]
    except:
        return None