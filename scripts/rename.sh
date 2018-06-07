png=.png
for i in `seq 120 199`;
do
mv img_$i$png img_$(($i-119))$png
done
