# GPGPU Performance and Power Modeling with DVFS

This repository contains the code for modeling/benchmarking NVIDIA GPU performance and power with dynamic voltage and frequency scaling. The relevant papers are as follows:
+ **Q. Wang** and X.-W. Chu, "GPGPU Performance Estimation with Core and Memory Frequency Scaling," IEEE International Conference on Parallel and Distributed Systems (ICPADS) 2018, Singapore, Dec 2018.[An extended journal version is under revew.]
+ **Q. Wang** and X.-W. Chu, "PP-G: Performance and Power Estimation of DVFS-enabled GPU,"(under preparation)

### Citation
```
@inproceedings{Wang2018perf, 
    author={Q. {Wang} and X. {Chu}}, 
    booktitle={2018 IEEE 24th International Conference on Parallel and Distributed Systems (ICPADS)}, 
    title={GPGPU Performance Estimation with Core and Memory Frequency Scaling}, 
    year={2018}, 
    pages={417-424}, 
    month={Dec},
}
```

## Content
1. Introduction
2. Usage
3. Results
4. Contacts

## Usage
### Dependencies and prerequisites
+ Python 2.7
+ CUDA 9.0 or above
+ NVIDIA GPU Driver (the latest version is recommended.)
+ OS requirement: Windows 7/10, Ubuntu 16.04 or above, CentOS 6/7.
+ Using "pip install -r requirements.txt" to install the required python libraries.

### Data Collection
To use the performance/power models, one should first collect the needed performance counters and the groundtruth of kernel execution time and average runtime power. There are two configuration files that users should edit. One is called the benchmark-setting file, stored in configs/benchmarks/, which defines how the benchmarks run and repeat, what performance counters to collect and what frequencies are tested. The other is called the kernel-setting file, stored in configs/kernels/, which defines the set of tested GPU applications.

We provide some examples in those two folders. For the benchmark-setting file, the parameters contain:
\[profile_control\]
+ iters: the number of repetitions for GPU kernels 
+ secs: the execution time that the GPU application should run by repeating the kernel, prior to "iters"
+ cuda_device_id: the GPU index under CUDA runtime API
+ nvIns_device_id: the GPU index under CUDA Driver, which might be different from "cuda_device_id"
+ rest_time: the time interval (s) between two consecutive benchmarks
+ power_sample_interval: the power sampling interval (ms)
+ metrics: the list of profiling performance counters
\[dvfs_control\]
+ coreF: the list of tested core frequencies
+ memF: the list of tested memory frequencies
+ powerState: the fixed power state controlled by the GPU driver 

The kernel-setting file is composed by a list of elements, each of which defines one GPU application. The format is:
```
[*application_name*]
args = ["*arguments*"]
kernels = ["*kernel_name*"]
```

## Contact
Email: [qiangwang@comp.hkbu.edu.hk](mainto:qiangwang@comp.hkbu.edu.hk)
Personal Website: [https://blackjack2015.github.io](https://blackjack2015.github.io)
Welcome any suggestion or concern!
