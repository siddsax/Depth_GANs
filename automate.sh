name=cGAN
png=.png
for i in `seq 3 3`;
do
   cp mappingData_2/s$i/* mappingData_2/val
   for j in `seq 1 3`;
   do
      cp mappingData_2/s$i/img_$(($j))$png mappingData_2/train
      cp mappingData_2/s$i/img_$(($j+20))$png mappingData_2/train
      DATA_ROOT=./mappingData_2 name=s$i$name$j lambda=100 alpha=1 which_direction=BtoA th train.lua
      DATA_ROOT=./mappingData_2 name=s$i$name$j lambda=100 alpha=1 which_direction=BtoA th test.lua
   done
   rm mappingData_2/train/*
   rm mappingData_2/val/*
done
