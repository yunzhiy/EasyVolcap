import os
import time
import torch
# from easyvolcap.engine import cfg
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from torch.profiler import profile, record_function, ProfilerActivity, schedule

def profiler_step():
    if 'profiler' in globals():
        # with without_live():
        profiler.step()
        # log('[PROF] Profiler stepped')


def profiler_start():
    if 'profiler' in globals():
        # with without_live():
        profiler.start()
        # log('[PROF] Profiler started')


def profiler_stop():
    if 'profiler' in globals():
        # with without_live():
        profiler.stop()
        # log('[PROF] Profiler stopped')


def profiler_enabled():
    return 'profiler' in globals()


def setup_profiler(enabled=False,
                   clear_previous=True,
                   skip_first=5,
                   wait=5,
                   warmup=5,
                   active=5,
                   repeat=3,
                   record_dir=f"data/record/hello",  # constructed in the same way
                   record_shapes=True,
                   profile_memory=True,
                   with_stack=True,
                   with_flops=True,
                   with_modules=True,
                   ):
    if enabled:
        log(yellow(f"[PROF] Profiling results will be saved to: {blue(record_dir)}"))
        if clear_previous:
            log(red(f'[PROF] Removing profiling result in: {blue(record_dir)}'))
            os.system(f'rm -rf {record_dir}')
        global profiler
        profiler = profile(schedule=schedule(skip_first=skip_first,
                                             wait=wait,
                                             warmup=warmup,
                                             active=active,
                                             repeat=repeat,
                                             ),
                           activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           on_trace_ready=torch.profiler.tensorboard_trace_handler(record_dir),
                           record_shapes=record_shapes,
                           profile_memory=profile_memory,
                           with_stack=with_stack,  # sometimes with_stack causes segmentation fault
                           with_flops=with_flops,
                           with_modules=with_modules
                           )

        # # MARK: modification of global config, is this good?
        # cfg.runner_cfg.epoch = 1
        # cfg.runner_cfg.ep_iter = skip_first + repeat * (wait + warmup + active)
        log('[PROF] Created profiler object')
