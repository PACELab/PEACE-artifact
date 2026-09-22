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
                 consecutive_exceed=5,
                 # ---- New knobs ----
                 inc_threshold=0.8,          # was 0.90; try 0.80–0.85 per your idea (1)
                 sweet_low=0.92,              # hysteresis band low bound (no-change if within [sweet_low, 1.00])
                 ramp_min_secs=1.5,           # min seconds between ANY freq change
                 violation_cooldown_secs=4.0, # after any >cap event in last window, disable increases
                 ema_alpha=0.4,               # EMA smoothing for power (0=off, 1=no smoothing); try 0.3–0.5
                 near_cap_band=0.97,          # when ratio >= this, clamp to very small up-steps
                 min_step_mhz=10,             # never step smaller than this (if your GPU accepts it)
                 snap_to_config=True):        # if gpu_clock_config provided, snap to nearest allowed
        self.args = args

        self.gpu_clocks = self.load_gpu_clocks(gpu_clock_config) if gpu_clock_config else None
        if self.gpu_clocks:
            self.gpu_clocks = sorted(int(x) for x in self.gpu_clocks)

        self.gpu_index = gpu_index
        self.power_cap = power_cap
        self.current_freq = start_freq
        self.min_freq = min(self.gpu_clocks) if self.gpu_clocks else min_freq
        self.max_freq = max(self.gpu_clocks) if self.gpu_clocks else max_freq
        self.adjust_step = adjust_step
        self.monitor_interval_ms = monitor_interval_ms
        self.consecutive_exceed = consecutive_exceed

        # New controls
        self.inc_threshold = inc_threshold
        self.sweet_low = sweet_low
        self.ramp_min_secs = ramp_min_secs
        self.violation_cooldown_secs = violation_cooldown_secs
        self.ema_alpha = ema_alpha
        self.near_cap_band = near_cap_band
        self.min_step_mhz = min_step_mhz
        self.snap_to_config = snap_to_config and (self.gpu_clocks is not None)

        self.set_gpu_persistent_mode()

        self.recent_powers = deque(maxlen=consecutive_exceed)
        self.last_adjust_ts = 0.0
        self.last_overcap_ts = -1e9
        self.power_ema = None

        # Initialize GPU frequency
        self.set_gpu_frequency(self.current_freq, self.current_freq)

        # logging (same as yours)
        log_level = logging.DEBUG if self.args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s [%(filename)s - %(funcName)s - line:%(lineno)d] - %(message)s',
            handlers=[logging.FileHandler(logfile)]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
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
    # ---------- NEW HELPERS ----------
    def _snap_up(self, freq):
        """Next allowed clock >= freq."""
        if not self.snap_to_config:
            return min(self.max_freq, freq)
        for f in self.gpu_clocks:
            if f >= freq:
                return min(f, self.max_freq)
        return self.max_freq

    def _snap_down(self, freq):
        """Next allowed clock <= freq."""
        if not self.snap_to_config:
            return max(self.min_freq, freq)
        prev = self.min_freq
        for f in self.gpu_clocks:
            if f > freq:
                return max(prev, self.min_freq)
            prev = f
        return max(prev, self.min_freq)

    def _cap_aware_upstep(self, ratio_below):
        """
        ratio_below = (inc_threshold - power_ratio) / inc_threshold in [0,1+]
        Far below -> bigger steps, near cap -> tiny steps.
        """
        # Base scale 0.3..2.0 of adjust_step depending on distance
        scale = 0.3 + 1.7 * max(0.0, min(1.5, ratio_below))  # cap at 1.5 for very low power
        step = int(max(self.min_step_mhz, self.adjust_step * scale))

        return step

    def _cap_aware_downstep(self, over_ratio):
        """
        over_ratio = (power_ratio - 1.0) in [0, +]
        If a little over -> moderate step; way over -> larger step.
        """
        scale = min(2.0, 1.0 + 2.0 * over_ratio)   # 1.0..2.0
        step = int(max(self.min_step_mhz, self.adjust_step * scale))
        return step
    # ---------- MODIFIED CONTROL LOOP ----------
    def monitor_and_adjust_once(self):
        result = self.read_gpu_power_and_freq()
        if result is None:
            return

        power_val, curr_freq, sm_percent = result

        # EMA smoothing
        if self.power_ema is None:
            self.power_ema = power_val
        else:
            self.power_ema = self.ema_alpha * power_val + (1 - self.ema_alpha) * self.power_ema

        self.recent_powers.append(self.power_ema)
        now = time.time()
        ratio = self.power_ema / self.power_cap

        # Track recent over-cap to gate future increases
        if ratio > 1.0:
            self.last_overcap_ts = now

#        self.logger.info(f"[GPUPowerMonitor] idx={self.gpu_index}, P={power_val:.1f}W (EMA {self.power_ema:.1f}), "
#                         f"cap={self.power_cap}W (r={ratio:.3f}), freq_set={self.current_freq} MHz, "
#                         f"freq_now={curr_freq} MHz, SM={sm_percent}%")
        self.logger.info(f"[GPUPowerMonitor] index= {self.gpu_index}, Power={power_val:.1f} W, Freq={self.current_freq} MHz, Current Freq={curr_freq} MHz, SM Utilization={sm_percent}%, EMA={self.power_ema:.1f}, r={ratio:.3f}")


        # Only adjust if we have enough samples AND we respect ramp rate
        if len(self.recent_powers) < self.consecutive_exceed:
            return
        if (now - self.last_adjust_ts) < self.ramp_min_secs:
            return

        # Low SM util? Skip adjustments to avoid chasing idle noise.
        if sm_percent <= 10:
            self.logger.debug("Low SM util; skipping adjustment.")
            self.recent_powers.clear()
            return

        # ---- Control logic with hysteresis + cap-aware steps ----
        # 1) Over cap -> decrease
        if ratio > 1.0:
            over_ratio = ratio - 1.0
            step = self._cap_aware_downstep(over_ratio)
            target = max(self.min_freq, self.current_freq - step)
            target = self._snap_down(target)
            if target != self.current_freq:
                self.logger.info(f"Over cap r={ratio:.3f} for {self.consecutive_exceed} reads: "
                                 f"down {self.current_freq}->{target} (step {self.current_freq - target} MHz)")
                self.current_freq = target
                self.set_gpu_frequency(target, target)
                self.last_adjust_ts = now
            self.recent_powers.clear()
            return

        # 2) Within sweet band -> hold
        if self.sweet_low <= ratio <= 1.0:
            self.logger.debug(f"In sweet band [{self.sweet_low:.2f},1.00], no change.")
            self.recent_powers.clear()
            return

        # 3) Under inc_threshold -> consider increase (but only if no recent violation)
        time_since_violation = now - self.last_overcap_ts
        if ratio < self.inc_threshold:
            if time_since_violation < self.violation_cooldown_secs:
                self.logger.debug(f"Under threshold but recent violation {time_since_violation:.1f}s ago; hold.")
                self.recent_powers.clear()
                return

            # Make step size shrink sharply when near cap
            if ratio >= self.near_cap_band:
                step = max(self.min_step_mhz, int(self.adjust_step * 0.25))
            else:
                # Distance-based scaling below inc_threshold
                headroom = max(0.0, self.inc_threshold - ratio)
                norm = headroom / self.inc_threshold  # 0..1+
                step = self._cap_aware_upstep(norm)

            target = min(self.max_freq, self.current_freq + step)
            target = self._snap_up(target)
            if target != self.current_freq:
                self.logger.info(f"Under inc_threshold r={ratio:.3f}: up {self.current_freq}->{target} "
                                 f"(step {target - self.current_freq} MHz)")
                self.current_freq = target
                self.set_gpu_frequency(target, target)
                self.last_adjust_ts = now
            self.recent_powers.clear()
            return

        # 4) Between inc_threshold and sweet_low -> gentle nudge upward only if zero violations recently
        if self.inc_threshold <= ratio < self.sweet_low:
            if time_since_violation < self.violation_cooldown_secs:
                self.logger.debug(f"Near sweet band but recent violation {time_since_violation:.1f}s; hold.")
                self.recent_powers.clear()
                return
            # very small step to creep up
            step = max(self.min_step_mhz, int(self.adjust_step * 0.2))
            target = self._snap_up(min(self.max_freq, self.current_freq + step))
            if target != self.current_freq:
                self.logger.info(f"Creeping toward sweet band: {self.current_freq}->{target} "
                                 f"(+{target - self.current_freq} MHz)")
                self.current_freq = target
                self.set_gpu_frequency(target, target)
                self.last_adjust_ts = now
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
