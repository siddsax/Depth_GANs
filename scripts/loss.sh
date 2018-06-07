name=_l10_e500_t
png=.png
for i in `seq 2 4`;
do
   for j in `seq 1 5`;
   do
       python loss.py s$i$name$j 
   done
done
