start nvidiaInspector.exe -showMonitoring
set app_folder=MemoryBenchmarking
set log_folder=Logs\memory
set conf_folder=Conf
set wt=5

nvidiaInspector.exe -forcepstate:0,16

for /f "tokens=*" %%c in (%conf_folder%\core_frequency.txt) do (
for /f "tokens=*" %%m in (%conf_folder%\memory_frequency.txt) do (

    ping /n %wt% 127.1>nul
    nvidiaInspector.exe -forcepstate:0,5 -setMemoryClock:0,1,%%m -setGpuClock:0,1,%%c

for /f "tokens=*" %%a in (%conf_folder%\memory_suite.txt) do (
    
    ping /n %wt% 127.1>nul
    echo sm_clock:%%c,mem_clock:%%m >> %log_folder%\%%a.txt
    %app_folder%\%%a >> %log_folder%\%%a.txt

)
)
)

ping /n %wt% 127.1>nul

nvidiaInspector.exe -forcepstate:0,16
