start nvidiaInspector.exe -showMonitoring
set app_folder=Applications
set log_folder=Logs\performance_power
set conf_folder=Conf
set conf_arg=Conf\arguments
set wt=2
set st=2
set devID=0

nvidiaInspector.exe -forcepstate:%devID%,16
::del /f /s /q %log_folder%\*.*

ping /n %wt% 127.1>nul

for /f "tokens=*" %%c in (%conf_folder%\core_frequency.txt) do (
for /f "tokens=*" %%m in (%conf_folder%\memory_frequency.txt) do (

    nvidiaInspector.exe -forcepstate:%devID%,5 -setMemoryClock:%devID%,1,%%m -setGpuClock:%devID%,1,%%c
    ping /n %wt% 127.1>nul

for /f "tokens=*" %%a in (%conf_folder%\applications.txt) do (
for /f "tokens=*" %%o in (%conf_arg%\%%a_args.txt) do (

    echo sm_clock:%%c,mem_clock:%%m >> dvfs.txt
    echo program:%%a,arguments:%%o >> dvfs.txt
    start /B nvidia-smi.exe -l %st% >> dvfs.txt
    ping /n %wt% 127.1>nul

    echo sm_clock:%%c,mem_clock:%%m >> %log_folder%\%%a.txt
    echo arguments:%%o >> %log_folder%\%%a.txt

    %app_folder%\%%a %%o >> %log_folder%\%%a.txt
    ping /n %wt% 127.1>nul

    tasklist|findstr "nvidia-smi.exe"
    taskkill /F /IM nvidia-smi.exe
    ping /n %wt% 127.1>nul

    nvprof %app_folder%\%%a %%o >> %log_folder%\%%a.txt 2>&1
    ping /n %wt% 127.1>nul

    nvprof --metrics achieved_occupancy,dram_read_transactions,dram_write_transactions,l2_read_transactions,shared_load_transactions,shared_store_transactions,cf_executed,ecc_transactions,eligible_warps_per_cycle,stall_memory_dependency %app_folder%\%%a %%o >> %log_folder%\%%a.txt 2>&1
    ping /n %wt% 127.1>nul
)
)
)
)

nvidiaInspector.exe -forcepstate:%devID%,16
ping /n %wt% 127.1>nul
pause

