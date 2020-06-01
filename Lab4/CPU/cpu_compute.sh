#~/bin/bash
clear
make
gcc -Wall -g parseResults.c -o parseResults

rm tmpResults
rm cpu_resultsTimes

M=12 #run computation for 12 times
echo "Compute script started."

echo "calculating uth.pgm"
echo "================== CPU uth.pgm ====================" >> cpu_resultsTimes
	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main uth.pgm output_uth.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> cpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> cpu_resultsTimes
echo "" >> cpu_resultsTimes
echo "done uth.pgm"

echo "calculating x_ray.pgm"
echo "================== CPU x_ray.pgm ====================" >> cpu_resultsTimes

	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main x_ray.pgm output_x_ray.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> cpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> cpu_resultsTimes
echo "" >> cpu_resultsTimes
echo "done x_ray.pgm"

echo "calculating ship.pgm"
echo "================== CPU ship.pgm ====================" >> cpu_resultsTimes

	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main ship.pgm output_ship.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> cpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> cpu_resultsTimes
echo "" >> cpu_resultsTimes
echo "done ship.pgm"

echo "calculating planet_surface.pgm"
echo "================== CPU planet_surface.pgm ====================" >> cpu_resultsTimes

	for j in `seq 1 $M`; 
	do
		results[$j]="$(./main planet_surface.pgm output_planet_surface.pgm)"
		all+="${results[$j]}"
		echo "result $j = ${results[$j]}"
	done

	echo "$all" > tmpResults
	./parseResults $M < tmpResults >> cpu_resultsTimes
	rm tmpResults
	all=""			
echo "===============================================" >> cpu_resultsTimes
echo "" >> cpu_resultsTimes
echo "done planet_surface.pgm"

echo "Compute script finished."