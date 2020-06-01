#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

__global__ void kernel_histogram(int *d_hist_out, unsigned char *d_img_in, int image_size) {
    __shared__ int shared_hist[BIN];

    if(threadIdx.x < BIN) {
        shared_hist[threadIdx.x] = 0;
    }

    __syncthreads();

    int pos = threadIdx.x + TIMES * (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x;
    
    #if (TIMES > 1)
		int i;
        for(i = 0; i < TIMES; i++) {
            if(pos < image_size) {
                atomicAdd(&shared_hist[d_img_in[pos]], 1);
            }
            pos += blockDim.x;
        }
    #else
        if(pos < image_size) {
            atomicAdd(&shared_hist[d_img_in[pos]], 1);
        }
    #endif

    __syncthreads();

    if(threadIdx.x < BIN)
        atomicAdd(&d_hist_out[threadIdx.x], shared_hist[threadIdx.x]);
}

__global__ void kernel_histogram_equalization(unsigned char * d_img_out, unsigned char *d_img_in, int *d_lut_in, int image_size) {
    __shared__ int shared_lut[BIN];

    int pos = threadIdx.x + TIMES * (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x;
    
    if(threadIdx.x < BIN) 
        shared_lut[threadIdx.x] = d_lut_in[threadIdx.x];

    __syncthreads();

    #if (TIME > 1)
		int i;
        for(i = 0; i < TIMES; i++) {
            if(pos < image_size){ 
                d_img_out[pos] = (unsigned char) shared_lut[d_img_in[pos]];
            }
            pos += blockDim.x;
        }
    #else
        if(pos < image_size) {
            d_img_out[pos] = (unsigned char) shared_lut[d_img_in[pos]];
        }
    #endif
}


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }   
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }  
    }
}

void lut_calculation(int *hist, int *lut, int image_size, int nbr_bin) {
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist[i++];
    }
    d = image_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*(BIN-1)/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        if(lut[i] > (BIN-1)){
            lut[i] = (BIN-1);
        } 
    }
}