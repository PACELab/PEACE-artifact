#!/usr/bin/env python3
"""
main.py

Example usage of the GPUPowerMonitor class.
"""

import argparse
from GPUpowerMonitor import GPUPowerMonitor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--power_cap", type=float, default=200.0)
    parser.add_argument("--start_freq", type=int, default=1300)
    parser.add_argument("--runtime", type=int, default=60,
                        help="How long (seconds) to run the monitoring loop.")

    args = parser.parse_args()

    # Create a GPUPowerMonitor instance
    monitor = GPUPowerMonitor(
        args=args,
        gpu_index=args.gpu_index,
        power_cap=args.power_cap,
        start_freq=args.start_freq,
        # you can also override other parameters here
    )

    # Suppose you already launched your Docker workloads here
    # ...
    # Then just run the monitor loop while tasks run
    monitor.run_monitor_loop(runtime=args.runtime)

    # After the loop, we can optionally do further checks or shut down Docker
    print("Done monitoring GPU power.")

if __name__ == "__main__":
    main()
