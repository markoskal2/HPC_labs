#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

#include <time.h>

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	
	struct timespec tv1, tv2;
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    histogram(hist, img_in.img, img_in.h * img_in.w, 256);
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
	printf("%10g \n", (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec));
	
	
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
	printf("%10g \n", (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec));
	
    return result;
}
