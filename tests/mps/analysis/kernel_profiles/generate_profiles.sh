
#ncu mem thrpughput in docker >0 -> profile with host
#export ncu
ncuFile=$1
nsysFile=$2
outputName=$3
#args error handle
if [ -z "$ncuFile" ] || [ -z "$nsysFile" ] || [ -z "$outputName" ]; then
    echo "Usage: $0 <ncuFile> <nsysFile> <outputName>"
    exit 1
fi
echo "ncu analysis started..."
ncu -i $ncuFile --csv   --print-details  all > source/ncu/"$outputName"_ncu.csv
echo "ncu analysis done. start nsys analysis..."
~/nsight-systems-2024.2.1/bin/nsys stats -r cuda_gpu_sum -o source/nsys/"$outputName"_nsys.csv  $nsysFile
echo "nsys analysis done"
echo "generating kernel profile..."
python get_kernel_profile.py source/ncu/"$outputName"_ncu.csv source/nsys/"$outputName"_nsys.csv_cuda_gpu_sum.csv $outputName