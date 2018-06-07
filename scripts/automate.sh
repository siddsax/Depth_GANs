name=_l1000_e500_a1_t
png=.png
for i in `seq 1 4`;
do
   cp mappingData_2/s$i/* mappingData_2/val
   for j in `seq 1 5`;
   do
      cp mappingData_2/s$i/img_$j$png mappingData_2/train
      DATA_ROOT=./mappingData_2 name=s$i$name$j lambda=1000 alpha=1 which_direction=AtoB th train.lua
      DATA_ROOT=./mappingData_2 name=s$i$name$j lambda=1000 alpha=1 which_direction=AtoB th test.lua
   done
   rm mappingData_2/train/*
   rm mappingData_2/val/*
done
