# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= main

#This is the compiler to use
CC = nvcc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -Xcompiler -Fast #-DCMD_LINE_INPUT=1

#These are the file dependencies
DEPENDENCIES = contrast-enhancement.cu histogram-equalization.cu

# make all will create all executables
all: $(EXECUTABLES)

# This is the rule to create any executable from the corresponding .cu 
# file with the same name.
%: %.cu
	$(CC) $(CFLAGS) $(DEPENDENCIES) $< -o $@
	gcc -Wall parseResults.c -o parseResults

# make clean will remove all executables, jpg and the 
# output of previous executions.
clean:
	rm -f $(EXECUTABLES) 
	rm -f parseResults
