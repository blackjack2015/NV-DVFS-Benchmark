# Release constraint on CLOCK adjustments
# P100

GPUID="${gpu:-0}"
FCORE="${fcore:-1328}"
FMEM="${fmem:-715}"

sudo nvidia-smi -i 0 -pl 250   # power limit
sudo nvidia-smi -i ${GPUID} -pm ENABLED   # Persistence mode ensures that the driver stays loaded even when no CUDA or X applications are running on the GPU
sudo nvidia-smi -i ${GPUID} -acp 0    # allow clock adjusting permission
sudo nvidia-smi -i ${GPUID} -ac ${FMEM},${FCORE}

# sudo nvidia-smi -i ${GPUID} --compute-mode=EXCLUSIVE_PROCESS
# sudo nvidia-smi -i 0 -ac 715,936
# sudo nvidia-smi -i 0 -ac 715,1037
# sudo nvidia-smi -i 0 -ac 715,1328
# sudo nvidia-smi -i 0 --compute-mode=EXCLUSIVE_PROCESS
# sudo nvidia-smi -i 0 -pl 150   # power limit
# sudo nvidia-smi -acp UNRESTRICTED -i 0 # allow non-admin users to change clocks




# Titan Xp
#sudo nvidia-smi -i 1 -pm 1
#sudo nvidia-smi -i 1 -acp 0
#sudo nvidia-smi -i 1 --compute-mode=EXCLUSIVE_PROCESS
#sudo nvidia-smi -i 1 -pl 150 

#sudo nvidia-smi -i 2 -pm 1
#sudo nvidia-smi -i 2 -acp 0
#sudo nvidia-smi -i 2 --compute-mode=EXCLUSIVE_PROCESS
#sudo nvidia-smi -i 2 -pl 180 

# sudo nvidia-smi -i 1 --auto-boost-default=DISABLED
# sudo nvidia-smi -i 1 --auto-boost-permission=UNRESTRICTED
# sudo nvidia-smi -i 0 -ac 810,1012

# Reset GPU
# sudo nvidia-smi -i 0 -pm 0
# sudo nvidia-smi -i 0 -e 0
# sudo nvidia-smi -i 0 -acp 1

# nvidia-smi -q -d CLOCK

#sudo nvidia-smi -i 0 -pm 1
#sudo nvidia-smi -i 0 -acp 0
#sudo nvidia-smi -i 0 --compute-mode=EXCLUSIVE_PROCESS
## sudo nvidia-smi -i 0 -ac 5005,139
## sudo nvidia-smi -i 0 -ac 5005,1911
#sudo nvidia-smi -i 0 -ac 5005,645
#sudo nvidia-smi -i 0 -pl 300
#
#sudo nvidia-smi -i 1 -pm 1
#sudo nvidia-smi -i 1 -acp 0
#sudo nvidia-smi -i 1 --compute-mode=EXCLUSIVE_PROCESS
## sudo nvidia-smi -i 1 -ac 3505,135
#sudo nvidia-smi -i 1 -ac 3505,1443
#sudo nvidia-smi -i 1 -ac 3505,444
#sudo nvidia-smi -i 1 -pl 165
#
#sudo nvidia-smi -i 2 -pm 1
#sudo nvidia-smi -i 2 -acp 0
#sudo nvidia-smi -i 2 --compute-mode=EXCLUSIVE_PROCESS
## sudo nvidia-smi -i 2 -ac 5705,139
## sudo nvidia-smi -i 2 -ac 5705,1911
#sudo nvidia-smi -i 2 -ac 5705,1417
#sudo nvidia-smi -i 2 -pl 300
#
