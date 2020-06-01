#~/bin/bash
clear

make
gcc -Wall parseResults.c -o parseResults

rm tmpResults
rm gpu_resultsTimes

M=12 #run computation for 12 times
echo "Compute script started."

echo "calculating uth.pgm"
echo "================== GPU uth.pgm ====================" >> gpu_resultsTimes

	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main ../Images/uth.pgm ../Images/gpu_output_uth.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> gpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> gpu_resultsTimes
echo "" >> gpu_resultsTimes
echo "done uth.pgm"

echo "calculating x_ray.pgm"
echo "================== GPU x_ray.pgm ====================" >> gpu_resultsTimes

	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main ../Images/x_ray.pgm ../Images/gpu_output_x_ray.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> gpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> gpu_resultsTimes
echo "" >> gpu_resultsTimes
echo "done x_ray.pgm"

echo "calculating ship.pgm"
echo "================== GPU ship.pgm ====================" >> gpu_resultsTimes

	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main ../Images/ship.pgm ../Images/gpu_output_ship.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> gpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> gpu_resultsTimes
echo "" >> gpu_resultsTimes
echo "done ship.pgm"

echo "calculating planet_surface.pgm"
echo "================== GPU planet_surface.pgm ====================" >> gpu_resultsTimes

	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main ../Images/planet_surface.pgm ../Images/gpu_output_planet_surface.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> gpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> gpu_resultsTimes
echo "" >> gpu_resultsTimes
echo "done planet_surface.pgm"

echo "Compute script finished."