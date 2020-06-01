#~/bin/bash
clear

make
icc -Wall parseResults.c -o parseResults

rm tmpResults
rm results

M=12 #run computation for 12 times
echo "Compute script started."

bash execute.sh $M 1 ./quake_seq input

bash execute.sh $M 1 ./quake input
bash execute.sh $M 4 ./quake input
bash execute.sh $M 8 ./quake input
bash execute.sh $M 14 ./quake input
bash execute.sh $M 28 ./quake input
bash execute.sh $M 56 ./quake input

echo "Compute script finished."
