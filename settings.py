
# Equation Type
DM_HID = 0	# cold miss latency is hidden
COMP_HID = 1    # compute/shared cycle is hidden
DM_COMP_HID = 2 # compute/shared cycle/cold miss is hidden
MEM_LAT_BOUND = 3 # memory latency cannot be hidden 
NO_HID = 4  # no hidden
MEM_HID = 5    # mem delay cycle is hidden
MIX = 6     # depend on frequency scaling

ABBRS = {"backpropForward": "BPFW", \
	 "backpropBackward": "BPBW", \
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
	 "matrixMulShared": "MMS", \
	 "matrixMulGlobal": "MMG", \
	 "mergeSort": "MS", \
	 "nn": "NN", \
	 "quasirandomGenerator": "quasiG", \
	 "reduction": "RD", \
	 "scalarProd": "SP", \
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
    def __init__(self):
        # Hardware Configuration
        self.a_L_DM = 222.78   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
        self.b_L_DM = 277.32   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
        self.L_L2 = 222   # 222 for gtx980
        self.L_INST = 4   # 4 for gtx980
        self.a_D_DM = 805.03    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.b_D_DM = 8.1762    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.D_L2 = 1     # 1 for l2 cache
        self.L_sh = 28    # 28 for gtx980
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 16 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 128 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 256 # 4096 for p100, 256 for gtx980, 384 for titanx

        self.DP_UNITS = 4
        self.SP_UNITS = 128
        self.SPEC_UNITS = 32
        self.LS_UNITS = 32

        self.CORE_FREQ = 1100
        self.MEM_FREQ = 3600
        
        # kernel equation type
        self.eqType = {}
        self.eqType['backprop'] = DM_HID    				# 0.111, too few workload
        self.eqType['BlackScholes'] = NO_HID
        self.eqType['conjugateGradient'] = DM_HID
        self.eqType['convolutionSeparable'] = COMP_HID
        self.eqType['convolutionTexture'] = NO_HID   
        self.eqType['fastWalshTransform'] = COMP_HID
        self.eqType['histogram'] = MEM_LAT_BOUND			# 0.076, to be witnessed
        self.eqType['hotspot'] = MEM_LAT_BOUND
        self.eqType['matrixMul'] = MEM_LAT_BOUND    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMulShared'] = MEM_LAT_BOUND    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMul(Global)'] = DM_COMP_HID
        self.eqType['matrixMulGlobal'] = DM_COMP_HID
        self.eqType['mergeSort'] = DM_COMP_HID 
        self.eqType['nn'] = NO_HID   				# 0.174, too few workload
        self.eqType['quasirandomGenerator'] = DM_COMP_HID
        self.eqType['reduction'] = MEM_LAT_BOUND
        self.eqType['scalarProd'] = DM_COMP_HID
        self.eqType['scan'] = DM_COMP_HID
        self.eqType['SobolQRNG'] = DM_COMP_HID
        self.eqType['sortingNetworks'] = DM_COMP_HID
        self.eqType['transpose'] = DM_HID
        self.eqType['vectorAdd'] = DM_HID

class TITANX:
    def __init__(self):
        # Hardware Configuration
        self.a_L_DM = 222.78   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
        self.b_L_DM = 500   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
        self.L_L2 = 222   # 222 for gtx980
        self.L_INST = 4   # 4 for gtx980
        self.a_D_DM = 1300    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.b_D_DM = 11.539    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.D_L2 = 1.2     # 1 for l2 cache
        self.L_sh = 28    # 28 for gtx980
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 28 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 128 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 384 # 4096 for p100, 256 for gtx980, 384 for titanx
        
        self.DP_UNITS = 4
        self.SP_UNITS = 128
        self.SPEC_UNITS = 32
        self.LS_UNITS = 32
        
        self.CORE_FREQ = 1800
        self.MEM_FREQ = 4500

        # kernel equation type
        self.eqType = {}
        self.eqType['backprop'] = MEM_LAT_BOUND   				# 0.111, too few workload
        self.eqType['BlackScholes'] = DM_COMP_HID
        self.eqType['conjugateGradient'] = DM_HID
        self.eqType['convolutionSeparable'] = DM_HID
        self.eqType['convolutionTexture'] = MEM_LAT_BOUND   
        self.eqType['fastWalshTransform'] = DM_HID
        self.eqType['histogram'] = MEM_LAT_BOUND			# 0.076, to be witnessed
        self.eqType['hotspot'] = MEM_LAT_BOUND
        self.eqType['matrixMul'] = MEM_LAT_BOUND    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMulShared'] = MEM_LAT_BOUND    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMul(Global)'] = NO_HID
        self.eqType['matrixMulGlobal'] = DM_COMP_HID
        self.eqType['mergeSort'] = NO_HID 
        self.eqType['nn'] = MEM_LAT_BOUND   		# 0.174, too few workload
        self.eqType['quasirandomGenerator'] = NO_HID
        self.eqType['reduction'] = MEM_LAT_BOUND
        self.eqType['scalarProd'] = DM_COMP_HID
        self.eqType['scan'] = DM_COMP_HID
        self.eqType['SobolQRNG'] = NO_HID
        self.eqType['sortingNetworks'] = DM_COMP_HID
        self.eqType['transpose'] = NO_HID
        self.eqType['vectorAdd'] = DM_HID

class P100:
    def __init__(self):
        # Hardware Configuration
        self.a_L_DM = 222.78 / 8   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
        self.b_L_DM = 277.32   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
        self.L_L2 = 263   # 222 for gtx980
        self.L_INST = 8   # 4 for gtx980
        self.a_D_DM = 805.03 / 4    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.b_D_DM = 1.5    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.D_L2 = 1     # 1 for l2 cache
        self.L_sh = 56    # 28 for gtx980
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 56 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 64 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 4096 # 4096 for p100, 256 for gtx980, 384 for titanx

        self.DP_UNITS = 32
        self.SP_UNITS = 64
        self.SPEC_UNITS = 16
        self.LS_UNITS = 16

        self.CORE_FREQ = 1328
        self.MEM_FREQ = 715

        # kernel equation type
        self.eqType = {}
        self.eqType['backprop'] = MEM_LAT_BOUND    				# 0.111, too few workload
        self.eqType['BlackScholes'] = NO_HID
        self.eqType['conjugateGradient'] = NO_HID
        self.eqType['convolutionSeparable'] = MEM_HID
        self.eqType['convolutionTexture'] = MEM_LAT_BOUND  
        self.eqType['fastWalshTransform'] = NO_HID
        self.eqType['histogram'] = NO_HID			# 0.076, to be witnessed
        self.eqType['hotspot'] = NO_HID
        self.eqType['matrixMul'] = NO_HID    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMulShared'] = NO_HID    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMul(Global)'] = COMP_HID
        self.eqType['matrixMulGlobal'] = COMP_HID
        self.eqType['mergeSort'] = NO_HID 
        self.eqType['nn'] = NO_HID   				# 0.174, too few workload
        self.eqType['quasirandomGenerator'] = MEM_LAT_BOUND
        self.eqType['reduction'] = MEM_HID
        self.eqType['scalarProd'] = NO_HID
        self.eqType['scan'] = DM_HID
        self.eqType['SobolQRNG'] = DM_HID
        self.eqType['sortingNetworks'] = COMP_HID
        self.eqType['transpose'] = DM_COMP_HID
        self.eqType['vectorAdd'] = DM_COMP_HID

