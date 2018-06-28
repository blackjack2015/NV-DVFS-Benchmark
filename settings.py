
# Equation Type
DM_HID = 0	# cold miss latency is hidden
COMP_HID = 1    # compute/shared cycle is hidden
DM_COMP_HID = 2 # compute/shared cycle/cold miss is hidden
MEM_LAT_BOUND = 3 # memory latency cannot be hidden 
NO_HID = 4  # no hidden
MIX = 5     # depend on frequency scaling

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
        self.eqType['matrixMul'] = NO_HID    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMulShared'] = NO_HID    			# 0.15, consider how to deal with shared memory
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
        self.b_L_DM = 277.32   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
        self.L_L2 = 222   # 222 for gtx980
        self.L_INST = 4   # 4 for gtx980
        self.a_D_DM = 805.03    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.b_D_DM = 8.1762    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
        self.D_L2 = 1     # 1 for l2 cache
        self.L_sh = 28    # 28 for gtx980
        self.WARPS_MAX = 64 # 64 for gtx980
        self.SM_COUNT = 28 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 128 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 384 # 4096 for p100, 256 for gtx980, 384 for titanx
        
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
        self.eqType['matrixMul'] = NO_HID    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMulShared'] = NO_HID    			# 0.15, consider how to deal with shared memory
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

class P100:
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
        self.SM_COUNT = 56 # 56 for p100, 16 for gtx980, 28 for titanx
        self.CORES_SM = 64 # 64 for p100, 128 for gtx980 and titanx
        self.WIDTH_MEM = 4096 # 4096 for p100, 256 for gtx980, 384 for titanx
        
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
        self.eqType['matrixMul'] = NO_HID    			# 0.15, consider how to deal with shared memory
        self.eqType['matrixMulShared'] = NO_HID    			# 0.15, consider how to deal with shared memory
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

