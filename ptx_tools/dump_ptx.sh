input_root=./applications/linux
output_root=./ptxs

bins=$(ls ${input_root})
for bin in ${bins}
do
    echo "cuobjdump ${input_root}/${bin} -arch sm_80 --dump-ptx 1>${output_root}/${bin}.ptx"
    cuobjdump ${input_root}/${bin} -arch sm_80 --dump-ptx 1>${output_root}/${bin}.ptx
done
