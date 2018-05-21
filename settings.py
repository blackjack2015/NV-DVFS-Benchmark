# gpu card and data file
gpucard = 'titanx'
csv_perf = "csvs/%s-DVFS-Performance.csv" % gpucard

# Equation Type
DM_HID = 0	# cold miss latency is hidden
COMP_HID = 1    # compute/shared cycle is hidden
DM_COMP_HID = 2 # compute/shared cycle/cold miss is hidden
MEM_LAT_BOUND = 3 # memory latency cannot be hidden 
NO_HID = 4  # no hidden
MIX = 5     # depend on frequency scaling

# Hardware Configuration
a_L_DM = 222.78   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
b_L_DM = 277.32   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
L_L2 = 222   # 222 for gtx980
L_INST = 4   # 4 for gtx980
a_D_DM = 805.03    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
b_D_DM = 8.1762 / 0.85   # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
a_D_DM = 17    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
b_D_DM = 22.8   # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
D_L2 = 1     # 1 for l2 cache
L_sh = 28    # 28 for gtx980
WARPS_MAX = 64 # 64 for gtx980
SM_COUNT = 28 # 56 for p100, 16 for gtx980, 28 for titanx
CORES_SM = 128 # 64 for p100, 128 for gtx980 and titanx
WIDTH_MEM = 256 # 4096 for p100, 256 for gtx980, 384 for titanx

# kernel equation type
eqType = {}
eqType['backprop'] = DM_HID    				# 0.111, too few workload
eqType['BlackScholes'] = NO_HID
eqType['conjugateGradient'] = DM_HID
eqType['convolutionSeparable'] = COMP_HID
eqType['convolutionTexture'] = NO_HID   
eqType['fastWalshTransform'] = COMP_HID
eqType['histogram'] = MEM_LAT_BOUND			# 0.076, to be witnessed
eqType['hotspot'] = MEM_LAT_BOUND
eqType['matrixMul'] = NO_HID    			# 0.15, consider how to deal with shared memory
eqType['matrixMulShared'] = NO_HID    			# 0.15, consider how to deal with shared memory
eqType['matrixMul(Global)'] = DM_COMP_HID
eqType['matrixMulGlobal'] = DM_COMP_HID
eqType['mergeSort'] = DM_COMP_HID 
eqType['nn'] = NO_HID   				# 0.174, too few workload
eqType['quasirandomGenerator'] = DM_COMP_HID
eqType['reduction'] = MEM_LAT_BOUND
eqType['scalarProd'] = DM_COMP_HID
eqType['scan'] = DM_COMP_HID
eqType['SobolQRNG'] = DM_COMP_HID
eqType['sortingNetworks'] = DM_COMP_HID
eqType['transpose'] = DM_HID
eqType['vectorAdd'] = NO_HID
