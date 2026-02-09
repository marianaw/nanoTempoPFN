#!/usr/bin/env python3
"""Test multi-device training locally by simulating 8 CPU devices."""

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'  # count=8 => 94 seconds, count=1 => > 2 minutes.

import jax
from train import main
from time import time

if __name__ == "__main__":
    # jax.distributed.initialize()
    # testing this in my laptop before burning cash in runpod
    print(f"Testing with {jax.local_device_count()} simulated devices")
    print(f"Devices: {jax.devices()}")
    start = time()
    main()
    end = time()
    print((end - start)/60.0)
