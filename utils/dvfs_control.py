import os


class DVFSController:

    def __init__(self, device_id):

        self.device_id = device_id

        # for linux
        # unlock the dvfs function
        os.system('sudo nvidia-smi -i %d -pl 300' % self.device_id)
        os.system('sudo nvidia-smi -i %d -pm ENABLED' % self.device_id)
        os.system('sudo nvidia-smi -i %d -acp 0' % self.device_id)

        # for windows nvidiaInspector
        # self.dvfs_cmd = 'nvidiaInspector.exe -setBaseClockOffset:%s,%d,%s -setMemoryClockOffset:%s,%d,%s' % (nvIns_dev_id, powerState, nvIns_dev_id, freqState, '%s', nvIns_dev_id, freqState, '%s')

    def set_frequency(self, core_freq, mem_freq):

        os.system('sudo nvidia-smi -i %d -lgc %d' % (self.device_id, core_freq))
        os.system('sudo nvidia-smi -i %d -lmc %d' % (self.device_id, mem_freq))
