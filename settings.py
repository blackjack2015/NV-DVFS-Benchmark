
# Equation Type
DM_HID = 0	# cold miss latency is hidden
COMP_HID = 1    # compute/shared cycle is hidden
DM_COMP_HID = 2 # compute/shared cycle/cold miss is hidden
MEM_LAT_BOUND = 3 # memory latency cannot be hidden 
NO_HID = 4  # no hidden
MEM_HID = 5    # mem delay cycle is hidden
MIX = 6     # depend on frequency scaling
COMP_BOUND = 7 # compute bound, compute delay
SHM_BOUND = 8 # shared memory delay

ABBRS = {"backpropForward": "BPFW", \
	 "backpropBackward": "BPBW", \
         "backprop": "BP", \
	 "binomialOptions": "BO", \
	 "BlackScholes": "BS", \
	 "cfd": "CFD", \
	 "conjugateGradient": "CG", \
	 "convolutionSeparable": "convS", \
	 "convolutionTexture": "convT", \
	 "dxtc": "DT", \
	 "eigenvalues": "EV", \
	 "fastWalshTransform": "FW", \
	 "gaussian": "GS", \
	 "histogram": "HIST", \
	 "hotspot": "HSP", \
         "matrixMul": "MMS", \
	 "matrixMulShared": "MMS", \
	 "matrixMul(Global)": "MMG", \
	 "matrixMulGlobal": "MMG", \
	 "mergeSort": "MS", \
	 "nn": "NN", \
	 "quasirandomGenerator": "quasiG", \
	 "reduction": "RD", \
	 "scalarProd": "SP", \
         "scan": "SC", \
	 "scanScanExclusiveShared": "SCEX", \
	 "scanUniformUpdate": "SCUU", \
	 "SobolQRNG": "SOR", \
	 "sortingNetworks": "SN", \
	 "transpose": "TR", \
	 "vectorAdd": "VA", \
	 "srad": "SRAD", \
	 "stereoDisparity": "SD", \
	 "pathfinder": "PF"}

def get_abbr(kernel):
    return ABBRS[kernel]

class GTX980:
    def __init__(self, dvfs_range = 'low'):
        # Hardware Configuration
        self.a_L_DM = 78.417   # a * f_core / f_mem + b
        self.b_L_DM = 97.62   # a * f_core / f_mem + b
        self.L_L2 = 222   # 222 for gtx980
        self.L_INST = 4   # 4 for gtx980
        self.a_D_DM = 805.03    # a / f_mem + b
        self.b_D_DM = 8.1762    # a / f_mem + b
        self.D_L2 = 1.2  # 1.2 for l2 cache, 20% inefficiency
        self.D_INST = 1     # 1.0 for compute throughput
        self.L_sh = 28    # 28 for gtx980
        self.D_sh = 1
        self.D_DP = 20
        self.D_TEX = 1
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 16 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 128 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 256 # 4096 for p100, 256 for gtx980, 384 for titanx

        self.TEX_UNITS = 8
        self.DP_UNITS = 4
        self.SP_UNITS = 128
        self.SPEC_UNITS = 32
        self.LS_UNITS = 32

        if dvfs_range == 'low':
            self.CORE_FREQ = 1000
            self.MEM_FREQ = 1000
            #self.CORE_FREQ = 500
            #self.MEM_FREQ = 500
        elif dvfs_range == 'high':
            self.CORE_FREQ = 1100
            self.MEM_FREQ = 3600
        
class GTX1080TI:
    def __init__(self):
        # Hardware Configuration
        self.a_L_DM = 178.22   # a * f_core / f_mem + b
        self.b_L_DM = 396.72   # a * f_core / f_mem + b
        self.a_D_DM = 7208.57    # a / f_mem + b
        self.b_D_DM = 10.0305    # a / f_mem + b
        self.L_L2 = 220   # 222 for gtx980
        self.D_L2 = 1.3     # 1.3 for l2 cache
        self.L_INST = 4   # 4 for gtx980
        self.D_INST = 1.2    # 1.2 for compute throughput
        self.L_sh = 28    # 28 for 1080ti
        self.D_sh = 0.95  # 1.0 is also OK. 
        self.D_DP = 20
        self.D_TEX = 1.4
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 28 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 128 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 384 # 4096 for p100, 256 for gtx980, 384 for titanx
        
        self.DP_UNITS = 4
        self.TEX_UNITS = 8
        self.SP_UNITS = 128
        self.SPEC_UNITS = 32
        self.LS_UNITS = 32
        
        self.CORE_FREQ = 2000
        self.MEM_FREQ = 5500
        #self.CORE_FREQ = 1600
        #self.MEM_FREQ = 4000

class P100:
    def __init__(self):
        # Hardware Configuration
        self.a_L_DM = 26.73   # a * f_core / f_mem + b 
        self.b_L_DM = 266.23   # a * f_core / f_mem + b
        self.L_L2 = 263   # 222 for gtx980
        self.L_INST = 8   # 4 for gtx980
        self.a_D_DM = 705.03 / 4    # a / f_mem + b 
        self.b_D_DM = 1.8  # a / f_mem + b
        self.D_L2 = 1.0     # 1 for l2 cache
        self.D_INST = 1.4    # 1.2 for compute throughput
        self.L_sh = 28    # 56 gives slicely better results.
        self.D_sh = 1
        self.D_DP = 1.2
        self.D_TEX = 2.5
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 56 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 64 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 4096 # 4096 for p100, 256 for gtx980, 384 for titanx

        self.TEX_UNITS = 4
        self.DP_UNITS = 32
        self.SP_UNITS = 64
        self.SPEC_UNITS = 16
        self.LS_UNITS = 16

        self.CORE_FREQ = 1328
        #self.CORE_FREQ = 607
        #self.CORE_FREQ = 1202
        self.MEM_FREQ = 715


class V100:
    def __init__(self):
        # Hardware Configuration
        self.a_L_DM = 26.73   # a * f_core / f_mem + b 
        self.b_L_DM = 133.23   # a * f_core / f_mem + b
        self.L_L2 = 263   # 222 for gtx980
        self.L_INST = 8   # 4 for gtx980
        self.a_D_DM = 1005.03 / 4    # a / f_mem + b
        self.b_D_DM = 2.5  # a / f_mem + b
        self.D_L2 = 1.0     # 1 for l2 cache
        self.D_INST = 1.2   # 1.1 for compute throughput
        self.L_sh = 28    # 56 for v100
        self.D_sh = 1
        self.D_DP = 1.2
        self.D_TEX = 2.5
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 80 # 80 for v100, 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 128 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 4096 # 4096 for p100, 256 for gtx980, 384 for titanx

        self.TEX_UNITS = 4
        self.DP_UNITS = 64
        self.SP_UNITS = 128
        self.SPEC_UNITS = 16
        self.LS_UNITS = 16

        self.CORE_FREQ = 1380
        #self.CORE_FREQ = 802
        self.MEM_FREQ = 877

