#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void getResults(float *table, char *msg, int N);

int main(int argc, char *argv[]){
	if(argc < 2 || argc > 3){
		printf("Error:argc < 2 || argc > 3\n");
		return -1;
	}
	
	int i;
	int N = atoi(argv[1]);
	float *results, *resultsHist, * resultsHistEq;
	
	results = (float *)malloc(sizeof(float)*N);
	resultsHist = (float *)malloc(sizeof(float)*N);
	resultsHistEq = (float *)malloc(sizeof(float)*N);
	
	if(N < 1){
		printf("Error: N < 2\n");
		return -1;
	}
		
	for(i=0;i<N;i++){
		scanf("%g",&(resultsHist[i]));
		scanf("%g",&(resultsHistEq[i]));
		scanf("%g",&(results[i]));
	}
	getResults(resultsHist,"GPU calc time(hist)", N);
	getResults(resultsHistEq,"GPU calc time(histEq)", N);
	getResults(results,"GPU total Time", N);
	
	free(results);
	free(resultsHist);
	free(resultsHistEq);
	
	return 0;
}

void getResults(float *table, char *msg, int N){
	int i, max_p, min_p;
	float max, min;
	max = table[0];
	min = table[0];
	max_p = 0;
	min_p = 0; 

	for(i=0;i<N;i++){
		if( table[i] > max ){
			max = table[i];
			max_p = i;
		}
		if( table[i] < min ){
			min = table[i];
			min_p = i;
		}
	}
	
	printf("%s:\n",msg);
	for(i=0;i<N;i++){
		if(i != max_p && i != min_p){
			printf("%g\n",table[i]);			
		}
	}
	printf("\n");
}
