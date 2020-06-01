#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[BIN], lut[BIN];
    int blockDimX, image_size;
    int gridDimX, gridDimY;

    int *d_hist, *d_lut;
    unsigned char *d_img_in, *d_img_out;
    image_size = img_in.w * img_in.h;

    struct timespec tv_begin, tv_end, tv1, tv2, tv3, tv4;

    /********Ksekina na metras xrono gia olokliri 
    tin ektelesi tou contrast_ench********/
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv_begin);

    gridDimX = 1;
    gridDimY = 1;
    if(image_size > THREADS_PER_BLOCK) {
    	blockDimX = THREADS_PER_BLOCK;
    	gridDimX = (image_size-1)/THREADS_PER_BLOCK/TIMES + 1;
    	if(gridDimX > 65535) {
    		gridDimY = gridDimX/65535 + 1;
    		gridDimX = 65535;
    	}
    }
    else {
		  blockDimX = image_size;
	}

    dim3 blockDim(blockDimX);
    dim3 gridDim(gridDimX, gridDimY);

    CUDAsafeCall(cudaMalloc(&d_hist, BIN*sizeof(int)));
    CUDAsafeCall(cudaMalloc(&d_lut, BIN*sizeof(int)));
    CUDAsafeCall(cudaMalloc(&d_img_in, image_size*sizeof(unsigned char)));
    CUDAsafeCall(cudaMalloc(&d_img_out, image_size*sizeof(unsigned char)));

    CUDAsafeCall(cudaMemset(d_hist, 0, BIN*sizeof(int)));
    CUDAsafeCall(cudaMemcpy(d_img_in, img_in.img, image_size*sizeof(unsigned char), cudaMemcpyHostToDevice));

    /********Ypologismos tou kernel istogrammatos********/
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

    kernel_histogram<<<gridDim, blockDim>>>(d_hist, d_img_in, image_size);
    CUDAsafeCall(cudaPeekAtLastError());
    CUDAsafeCall(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    /********Ypologismos tou kernel istogrammatos********/

    printf("%10g \n", (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec));

    CUDAsafeCall(cudaMemcpy(hist, d_hist, BIN*sizeof(int), cudaMemcpyDeviceToHost));

    /********Ypologismos tou lookup table gia ton histogram_equalization********/
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    lut_calculation(hist, lut, image_size, BIN);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    /********Ypologismos tou lookup table gia ton histogram_equalization********/

    CUDAsafeCall(cudaMemcpy(d_lut, lut, BIN * sizeof(int), cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv3);
    kernel_histogram_equalization<<<gridDim, blockDim>>>(d_img_out, d_img_in, d_lut, image_size);

    CUDAsafeCall(cudaPeekAtLastError());
    CUDAsafeCall(cudaDeviceSynchronize());
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv4);

    /********Ypologismos tou kernel histogram_equalization mesa
    ston ypologismo yparxei kai o xronos ypologismou tou lut
    kai tis metaforas twn dedomenwn apo to host sto device********/

    printf("%10g \n", ((double)(tv2.tv_nsec - tv1.tv_nsec)/1000000000.0 + (double)(tv2.tv_sec - tv1.tv_sec)) +
    	((double)(tv4.tv_nsec - tv3.tv_nsec)/1000000000.0 + (double)(tv4.tv_sec - tv3.tv_sec)));


    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));

    CUDAsafeCall(cudaMemcpy(result.img, d_img_out, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv_end);
    /********Stamata to teliko roloi gia tous ypologismous********/

    printf("%10g \n", (double) (tv_end.tv_nsec - tv_begin.tv_nsec) / 1000000000.0 + (double) (tv_end.tv_sec - tv_begin.tv_sec));

#ifdef CPU_COMPARISON
    PGM_IMG cpu_result;
    int cpu_hist[BIN];

    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
    cpu_result.w = img_in.w;
    cpu_result.h = img_in.h;
    cpu_result.img = (unsigned char *) malloc(cpu_result.w * cpu_result.h * sizeof(unsigned char));


    histogram(cpu_hist, img_in.img, img_in.h * img_in.w, BIN);
    histogram_equalization(cpu_result.img, img_in.img, cpu_hist, cpu_result.w * cpu_result.h, BIN);
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

   	printf ("CPU total time (hist_equal) = %10g seconds\n\n",
            (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
            (double) (tv2.tv_sec - tv1.tv_sec));

   	unsigned int i;
   	int diff = 0;

   	for(i = 0; i < BIN; i++)
   		if(cpu_hist[i] != hist[i]) diff = 1;

   	if(!diff)
   		printf("CPU and GPU histogram match.");
   	else 
   		printf("CPU and GPU histogram don't match.");

   	diff = 0;
   	for(i = 0; i < image_size; i++)
   		if(cpu_result.img[i] != result[i]) diff = 1;

   	if(!diff)
   		printf("CPU and GPU result image match.");
   	else
   		printf("CPU and GPU result image don't match.");

   	free(cpu_result.img);
#endif

   	CUDAsafeCall(cudaFree(d_hist));
   	CUDAsafeCall(cudaFree(d_lut));
   	CUDAsafeCall(cudaFree(d_img_in));
   	CUDAsafeCall(cudaFree(d_img_out));

   	CUDAsafeCall(cudaDeviceReset());

    return result;
}
