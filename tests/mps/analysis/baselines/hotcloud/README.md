# Baseline - Hotcloud (Xu et al) feature set guide
## get Avg CPU Mem metrics
```
#autoregressive
bash record_cpu_autoregressive.sh
#all other models
bash record_cpu.sh

#parse cpu mem metrics
python hotcloud_parse_cpu_mem.py
```

# get kernel metrics
```
#modify input, output dir
bash hotcloud_parse_ncu_kernels.sh
```