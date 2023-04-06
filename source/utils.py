import json

import numpy as np


class MSEAverageMeter:
    def __init__(self, ndim, retain_axis, n_values=3):
        """
        Calculate average without overflows
        :param ndim: Number of dimensions
        :param retain_axis: Dimension to get average along
        :param n_values: Number of values along retain_axis
        """
        self.count = 0
        self.average = np.zeros(n_values, dtype=np.float64)
        self.retain_axis = retain_axis
        self.targets = []
        self.predictions = []
        self.axis = tuple(np.setdiff1d(np.arange(0, ndim), retain_axis))

    def add(self, pred, targ):
        self.targets.append(targ)
        self.predictions.append(pred)
        val = np.average((targ - pred) ** 2, axis=self.axis)
        c = np.prod([targ.shape[i] for i in self.axis])
        ct = c + self.count
        self.average = self.average * (self.count / ct) + val * (c / ct)
        self.count = ct

    def get_channel_avg(self):
        return self.average

    def get_total_avg(self):
        return np.average(self.average)

    def get_elements(self, axis):
        return np.concatenate(self.predictions, axis=axis), np.concatenate(self.targets, axis=axis)


def load_config(default_config, args, unknown_args):
    """
    Combine the arguments passed by user with configuration file given by user [and/or] default configuration. Convert extra named arguments to correct format.
    :param default_config: path to file
    :param args: known arguments passed by user
    :param unknown_args: unknown arguments passed by user
    :return: known_arguments, unknown_arguments
    """
    kwargs = {}

    def convert_value(y):
        try:
            return int(y)
        except:
            pass
        try:
            return float(y)
        except:
            pass
        if y == 'True' or y == 'False':
            return y == 'True'
        else:
            return y

    def convert_arrry(x):
        if not x:
            return True
        elif len(x) == 1:
            return x[0]
        return x

    i = 0
    while i < len(unknown_args):
        if unknown_args[i].startswith('--'):
            token = unknown_args[i].lstrip('-')
            options = []
            i += 1
            while i < len(unknown_args) and not unknown_args[i].startswith('--'):
                options.append(convert_value(unknown_args[i]))
                i += 1
            kwargs[token] = convert_arrry(options)

    if 'config' in kwargs:
        args.config = kwargs['config']
        del kwargs['config']
    with open(args.config, 'r') as f:
        config = json.load(f)

    values = vars(args)

    def add_missing_config(dictionary, remove=False):
        for key in values:
            if values[key] in [None, False] and key in dictionary:
                values[key] = dictionary[key]
                if remove:
                    del dictionary[key]

    add_missing_config(kwargs, True)        # specified args listed as unknowns
    add_missing_config(config)              # configuration from file for unspecified variables
    if args.config != default_config:       # default config
        with open(default_config, 'r') as f:
            default_configs = json.load(f)
        add_missing_config(default_configs)

    try:
        if args.channels is not None and type(args.channels) is str:
            args.channels = [int(i) for i in args.channels.split(',')]
    except:
        pass

    if 'kwargs' in config:
        kwargs = {**config['kwargs'], **kwargs}

    return args, kwargs

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)


def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B * T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B * T,
        size=int(B * T * p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res


def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")


def init_dl_program(
        device_name,
        seed=None,
        use_cudnn=True,
        deterministic=False,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]

