start nvidiaInspector.exe -showMonitoring
set app_folder=Applications
set log_folder=Logs\power
set conf_folder=Conf
set conf_arg=Conf\arguments\power
set wt=15
set st=3
set devID=0

nvidiaInspector.exe -forcepstate:%devID%,16

ping /n %wt% 127.1>nul

for /f "tokens=*" %%c in (%conf_folder%\core_frequency.txt) do (
for /f "tokens=*" %%m in (%conf_folder%\memory_frequency.txt) do (

    ping /n %wt% 127.1>nul
    nvidiaInspector.exe -forcepstate:%devID%,5 -setMemoryClock:%devID%,1,%%m -setGpuClock:%devID%,1,%%c

for /f "tokens=*" %%a in (%conf_folder%\applications_power.txt) do (
for /f "tokens=*" %%o in (%conf_arg%\%%a_args.txt) do (

    start /B nvidia-smi.exe -l %st% >> dvfs.txt
    ping /n %wt% 127.1>nul

    echo sm_clock:%%c,mem_clock:%%m >> %log_folder%\%%a.txt
    echo arguments:%%o >> %log_folder%\%%a.txt

    %app_folder%\%%a %%o >> %log_folder%\%%a.txt
    ping /n %wt% 127.1>nul
    tasklist|findstr "nvidia-smi.exe"
    taskkill /F /IM nvidia-smi.exe

)
)
)
)

ping /n %wt% 127.1>nul

