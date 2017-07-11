# NV-DVFS-Benchmark

This is the tools for DVFS experiments for NVIDIA GPU. Running the script "DVFS_Benchmark.bat" and then you will obtain all the energy information of the applications listed in the folder "Applications".

The folder "data" includes the data used when applications is launched.
The folder "Applications" includes the executable files of GPU applications together with needed library.
The folder "Conf" includes the runtime settings such as memory frequency, core frequency and benchmarking target applications.
The folder "Logs" includes the executing logs of all the applications under different DVFS settings.

Using wt to set the time interval between two consecutive benchmark sample.
Using st to set the power sampling interval.

We use nvidiaInspector to adjust the core frequency and memory frequency.The available adjustments may vary from different GPUs. One can test the range of voltage as well as frequency before running this benchmark.