
for i in {1..2048}
do

printf "$i  " >> logs.txt

perf stat -e L1-dcache-loads,L1-dcache-load-misses ./build/AIInference ${i} 2>&1 | grep -oP ".{5}%" >> logs.txt 

done

