#!/usr/bin/env python3
"""
gpu_power_monitor.py

A small module defining a GPUPowerMonitor class that:
- Adjusts GPU frequency
- Reads GPU power
- Monitors power usage periodically
"""

import subprocess
import time
import numpy as np
from collections import deque
import logging

class GPUPowerMonitor:
    def __init__(self, args,
                 logfile,
                 gpu_index=1,
                 power_cap=200.0,
                 start_freq=1300,
                 gpu_clock_config=None,
                 min_freq=1530,
                 max_freq=1530,
                 adjust_step=100,
                 monitor_interval_ms=500,
                 consecutive_exceed=5):
        """
        :param gpu_index:          GPU index to operate on
        :param power_cap:          Power cap in Watts
        :param start_freq:         Initial clock frequency (MHz)
        :param min_freq:           Minimum allowed clock frequency (MHz)
        :param max_freq:           Maximum allowed clock frequency (MHz)
        :param adjust_step:        Step (MHz) to increase/decrease freq
        :param monitor_interval_ms Interval (ms) between power reads
        :param consecutive_exceed: # of consecutive reads above the cap to trigger freq decrease
        """
        self.args = args
        
        self.gpu_clocks = self.load_gpu_clocks(gpu_clock_config) if gpu_clock_config else None
        
        self.gpu_index = gpu_index
        self.power_cap = power_cap
        self.current_freq = start_freq
        self.min_freq = min(self.gpu_clocks) if gpu_clock_config else min_freq
        self.max_freq = max(self.gpu_clocks) if gpu_clock_config else max_freq
        #the next best clock within clock config
        self.adjust_step = adjust_step
        self.monitor_interval_ms = monitor_interval_ms
        self.consecutive_exceed = consecutive_exceed

        #set gpu to persistent mode
        self.set_gpu_persistent_mode()
        
        # A deque to hold recent power readings
        self.recent_powers = deque(maxlen=consecutive_exceed)

        # Initialize GPU frequency
        self.set_gpu_frequency(self.current_freq, self.current_freq)

        #Set up *global* logging if not configured elsewhere
        log_level = logging.DEBUG if self.args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s [%(filename)s - %(funcName)s - line:%(lineno)d] - %(message)s',
            handlers=[
                #logging.StreamHandler(),
                logging.FileHandler(logfile)
            ]
        )

        # 2) Grab a logger specifically for this class.
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("Logger for GPUPowerMonitor initialized.")
        self.logger.info(f"args used for GPUPowerMonitor: {args}")

    
    def load_gpu_clocks(self, gpu_clock_config):
        """
        Loads GPU clock frequency configurations from a JSON file.
        """
        try:
            import json
            with open(gpu_clock_config, 'r') as f:
                gpu_clocks = json.load(f)
            return list(gpu_clocks)
        except Exception as e:
            print(f"[GPUPowerMonitor] Error loading GPU clocks: {e}")
            return None

    def set_gpu_persistent_mode(self):
        """
        Sets GPU to persistence mode using nvidia-smi (requires sudo).
        """
        cmd = f"sudo nvidia-smi -i {self.gpu_index} -pm 1"
        print(f"[GPUPowerMonitor] Setting GPU persistence mode: {cmd}")
        #wait for subprocess to cmd to finish before continuing

        subprocess.run(cmd, shell=True, check=False)
    


    def set_gpu_frequency(self, min_freq, max_freq):
        """
        Sets the GPU clock frequency range using nvidia-smi (requires sudo).
        """
        cmd = f"sudo nvidia-smi -i {self.gpu_index} -lgc {min_freq},{max_freq}"
        print(f"[GPUPowerMonitor] Setting GPU frequency: {cmd}")
        subprocess.run(cmd, shell=True, check=False)

    def read_gpu_power_and_freq(self):
        """
        Reads instantaneous power draw (Watts), clock frequency, and SM utilization for self.gpu_index via nvidia-smi.
        Returns tuple of (power, clock, sm_utilization) or None if error.
        """
        cmd = (f"nvidia-smi -i {self.gpu_index} --query-gpu=power.draw,clocks.sm,utilization.gpu "
              f"--format=csv,noheader,nounits")
        try:
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            values = output.split(', ')
            if len(values) == 3:
                return float(values[0]), int(values[1]), int(values[2])
            print(f"[GPUPowerMonitor] Error: Expected 3 values but got {len(values)}: {values}")
            return None
        except Exception as e:
            print(f"[GPUPowerMonitor] Error reading GPU metrics: {e}")
            return None

    def monitor_and_adjust_once(self):
        """
        - Reads current GPU power
        - Appends to recent power readings
        - Dynamically adjusts frequency based on power difference from cap
        - Only adjusts frequency if SM utilization is above 15%
        """
        result = self.read_gpu_power_and_freq()
        if result is None:
            # Could not read metrics
            return
        
        power_val, curr_freq, sm_percent = result
        self.recent_powers.append(power_val)
        self.logger.info(f"[GPUPowerMonitor] index= {self.gpu_index}, Power={power_val:.1f} W, Freq={self.current_freq} MHz, Current Freq={curr_freq} MHz, SM Utilization={sm_percent}%")

        # If we have enough consecutive readings
        if len(self.recent_powers) == self.consecutive_exceed:
            # Check if all exceed power 
            self.logger.debug(f"self recent powers: {self.recent_powers}")

            # Only adjust frequency if SM utilization is above 15%
            if sm_percent <= 10:
                self.logger.debug(f"[GPUPowerMonitor] SM Utilization is {sm_percent}%, below 15% threshold. Skipping frequency adjustment.")
                self.recent_powers.clear()
                return

            avg_power = sum(self.recent_powers) / len(self.recent_powers)
            power_diff_ratio = avg_power / self.power_cap
            
            # Calculate dynamic adjustment factor (0.5 to 2.0) based on difference
            if power_diff_ratio > 1.0:  # Power exceeds cap
                # More aggressive adjustment when power is much higher than cap
                adjustment_factor = min(1.5, power_diff_ratio)
                dynamic_step = int(self.adjust_step * adjustment_factor)
                new_freq = max(self.min_freq, self.current_freq - dynamic_step)
                
                if new_freq != self.current_freq:
                    self.logger.info(f"[GPUPowerMonitor] Power > cap (ratio: {power_diff_ratio:.2f}) for last {self.consecutive_exceed} readings. "
                        f"Decreasing freq from {self.current_freq} to {new_freq} (step: {dynamic_step} MHz)")
                    self.current_freq = new_freq
                    self.set_gpu_frequency(self.current_freq, self.current_freq)
                self.recent_powers.clear()
                
            # Under power threshold (use different thresholds for fine-grained control)
            elif power_diff_ratio < 0.90:  # Power is under 90% of cap
                # Calculate how far below the threshold we are (0.0 to 0.9)
                headroom_ratio = (0.90 - power_diff_ratio) / 0.90
                
                # More conservative adjustment when close to cap, more aggressive when far below
                adjustment_factor = 0.5 + (headroom_ratio * 1.5)  # Range from 0.5 to 2.0
                dynamic_step = int(self.adjust_step * adjustment_factor)
                
                new_freq = min(self.max_freq, self.current_freq + dynamic_step)
                
                if new_freq != self.current_freq:
                    self.logger.info(f"[GPUPowerMonitor] Power < 90% of cap (ratio: {power_diff_ratio:.2f}) for last {self.consecutive_exceed} readings. "
                        f"Increasing freq from {self.current_freq} to {new_freq} (step: {dynamic_step} MHz)")
                    self.current_freq = new_freq
                    self.set_gpu_frequency(self.current_freq, self.current_freq)
                self.recent_powers.clear()
                
            # Between 90% and 100% of cap - the sweet spot, no adjustment needed
            else:
                self.logger.debug(f"[GPUPowerMonitor] Power within optimal range ({power_diff_ratio:.2f} of cap). No adjustment needed.")
                self.recent_powers.clear()

    def run_monitor_loop(self, runtime=60):
        """
        A simple blocking loop that runs for 'runtime' seconds.
        Every interval, we read power and adjust freq if needed.
        """
        start = time.time()
        stop = start + runtime

        while time.time() < stop:
            self.monitor_and_adjust_once()
            time.sleep(self.monitor_interval_ms / 1000.0)
        
        # Optionally restore default freq
        self.restore_default_freq()

    def restore_default_freq(self):
        """
        Restore default GPU frequency settings after we’re done.
        """
        cmd = f"sudo nvidia-smi -i {self.gpu_index} -rgc"
        print("[GPUPowerMonitor] Restoring default GPU freq settings.")
        subprocess.run(cmd, shell=True, check=False)


class GPUStaticFreqMonitor(GPUPowerMonitor):
    def __init__(self, args, logfile, freq_list, durations=None, **kwargs):
        """
        Initialize GPUStaticFreqMonitor with a list of frequencies and durations.
        
        :param freq_list: List of GPU frequencies to test (e.g., [300, 900])
        :param durations: List of durations (seconds) for each frequency, or a single duration for all
        :param kwargs: Additional arguments passed to GPUPowerMonitor
        """
        super().__init__(args, logfile, **kwargs)
        self.freq_list = sorted(freq_list, reverse=False)  # Sort in descending order
        
        
        # Handle durations
        if durations is None:
            self.durations = [60] * len(freq_list)  # Default: 60 seconds per frequency
        elif isinstance(durations, (int, float)):
            self.durations = [durations] * len(freq_list)  # Single duration for all
        elif isinstance(durations, list):
            if len(durations) != len(freq_list):
                raise ValueError("Length of durations must match length of freq_list")
            self.durations = durations
        else:
            raise ValueError("durations must be a number or a list of numbers")
        
        self.logger.info(f"Initialized GPUStaticFreqMonitor with freq_list={self.freq_list}, durations={self.durations}")

    def run_monitor_loop(self, runtime, initial_duration=30):
        """
        Run the monitor loop with the following schedule:
        - Run max GPU frequency (1530 MHz) for initial_duration (30 seconds)
        - Run each frequency in freq_list (highest to lowest) for its specified duration
        Log timestamp when frequency changes.
        
        :param initial_duration: Duration (seconds) for initial max frequency run
        """
        # Step 1: Run max GPU frequency (1530 MHz) for 30 seconds
        self.current_freq = self.args.max_freq
        self.set_gpu_frequency(self.current_freq, self.current_freq)
        self.logger.info(f"Starting with max GPU frequency: {self.current_freq} MHz")
        
        #sleep for initial_duration
        self.logger.info(f"Running at max frequency for {initial_duration} seconds")
        time.sleep(initial_duration)
        
        
        
        # Step 2: Run each frequency in the provided list
        for freq, duration in zip(self.freq_list, self.durations):
            self.current_freq = freq
            self.set_gpu_frequency(self.current_freq, self.current_freq)
            
            #sleep for duration
            self.logger.info(f"Running at {self.current_freq} MHz for {duration} seconds")
            time.sleep(duration)
            
        #log the current freq
        self.logger.info(f"Final GPU frequency: {self.current_freq} MHz")
        

if __name__ == "__main__":
    # Example usage
    # Logging Configuration
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--logfile", type=str, default="gpu_power_monitor.log", help="Log file path")
    parser.add_argument("--gpu_index", type=int, default=1)
    parser.add_argument("--power_cap", type=float, default=200.0)
    parser.add_argument("--start_freq", type=int, default=1300)
    parser.add_argument("--runtime", type=int, default=9999999999999,
                        help="How long (seconds) to run the monitoring loop.")
    parser.add_argument("--gpu_clock_config", type=str, default=None,
                        help="Path to JSON file with GPU clock configurations.")  
    parser.add_argument("--min_freq", type=int, default=300)
    parser.add_argument("--max_freq", type=int, default=1530)
    parser.add_argument("--adjust_step", type=int, default=100)
    parser.add_argument("--monitor_interval_ms", type=int, default=500)
    parser.add_argument("--consecutive_exceed", type=int, default=5)
    #add args for GPUStaticFreqMonitor
    parser.add_argument("--static_freq", action="store_true")
    #args for GPUStaticFreqMonitor only
    parser.add_argument("--freq_list", type=int, nargs='+', default=[300, 900],
                        help="List of GPU frequencies to test (e.g., --freq_list 300 900)")
    parser.add_argument("--durations", type=float, nargs='*', default=120,
                        help="Duration (seconds) for each frequency, or a single duration for all")
    args = parser.parse_args()
    
    if args.static_freq:
        monitor = GPUStaticFreqMonitor(
        args,
        logfile=args.logfile,
        gpu_index=args.gpu_index,
        gpu_clock_config=args.gpu_clock_config,
        power_cap=args.power_cap,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        adjust_step=args.adjust_step,
        start_freq=args.start_freq,
        monitor_interval_ms=args.monitor_interval_ms,
        consecutive_exceed=args.consecutive_exceed,
        freq_list=args.freq_list,
        durations=args.durations
        )
    else:
        monitor = GPUPowerMonitor(args, logfile=args.logfile, gpu_index=args.gpu_index,gpu_clock_config=None ,power_cap=args.power_cap, 
                                    min_freq=args.min_freq, max_freq=args.max_freq, adjust_step=args.adjust_step, start_freq=args.start_freq,
                                    monitor_interval_ms=args.monitor_interval_ms, consecutive_exceed=args.consecutive_exceed)
    monitor.run_monitor_loop(runtime=args.runtime)
    print("Done monitoring GPU power.")
