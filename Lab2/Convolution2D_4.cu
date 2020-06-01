/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define maxAccuracy  	0.000005
#define minAccuracy	5.00

#define COMMAND_INPUT
#define TIME_CALC
#define ACC

/*******************CUDA Error Check*******************/
#define CUDAsafeCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDAcheckError() __cudaCheckError(__FILE__, __LINE__)

inline void gpuAssert(cudaError_t err, const char *file, int line, bool abort = true) {
    
    if(err != cudaSuccess) {
	fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(err), file, line);
	if(abort)
	    exit(err);
    }
}

inline void __cudaCheckError(const char *file, const int line) {
    
    cudaError error = cudaGetLastError();
    if(cudaSuccess != error) {
	fprintf(stderr, "CUDAcheckError failed at %s: %i: %s\n", file, line, cudaGetErrorString(error));
	exit(-1);
    }
    
    return; //an ola pane kala
}
/*******************CUDA Error Check*******************/

////////////////////////////////////////////////////////////////////////////////
// GPU row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
		       int imageW, int imageH, int filterR) {
    
    int blockID = (gridDim.x * blockIdx.y) + (gridDim.x * gridDim.y * blockIdx.z) + blockIdx.x;
    
    int threadID = threadIdx.x + (blockID * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x);
    
    int x = threadID % imageH;
    int y = threadID / imageH;
    int k, d;
    
    float sum = 0;
    
    for (k = -filterR; k <= filterR; k++) {
	d = x + k;
	
	if (d >= 0 && d < imageW) {
	    sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
	}     
	
	d_Dst[y * imageH + x] = sum;
    }
    
}

////////////////////////////////////////////////////////////////////////////////
// GPU column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
				     int imageW, int imageH, int filterR) {
    
    int blockID = (gridDim.x * blockIdx.y) + (gridDim.x * gridDim.y * blockIdx.z) + blockIdx.x;
    
    int threadID = threadIdx.x + (blockID * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x);
    
    int x = threadID % imageH;
    int y = threadID / imageH;
    int k, d;
    
    float sum = 0;
    
    for (k = -filterR; k <= filterR; k++) {
	d = y + k;
	
	if (d >= 0 && d < imageH) {
	    sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
	}
	
	d_Dst[y * imageH + x] = sum;
    }
    
				     }
				     


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}


////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU;


    int imageW;
    int imageH;
    unsigned int i;
    
#ifdef COMMAND_INPUT
    
    if(argc != 3) {
	printf("Error with command line args.");
	exit(1);
    }
    filter_radius = (unsigned int) atoi(argv[1]);
    imageW = atoi(argv[2]);
    
#else
    
    printf("Enter filter radius : ");
    scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    
#endif
    
    imageH = imageW;
    printf("Filter Radius: %i", filter_radius);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    //printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    
    /*****************************CUDA*****************************/
    
    float
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU,
    *h_OutputGPU;
    
    h_OutputGPU = (float *) malloc(imageW * imageH * sizeof(float));
     
    CUDAsafeCall(cudaMalloc((void **) &d_Filter, FILTER_LENGTH * sizeof(float)));
    CUDAsafeCall(cudaMalloc((void **) &d_Input, imageW * imageH * sizeof(float)));
    CUDAsafeCall(cudaMalloc((void **) &d_Buffer, imageW * imageH * sizeof(float)));
    CUDAsafeCall(cudaMalloc((void **) &d_OutputGPU, imageW * imageH * sizeof(float)));
    
    /**************************************************************/
    
    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < (unsigned int) imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }
    
    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");
#ifdef TIME_CALC
    //ksekina to roloi gia ti metrisi xronou sti CPU
    struct timespec tv1, tv2;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
#endif
    
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

#ifdef TIME_CALC
    //exoume parei tin metrisi gia to xrono tis CPU
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    
#ifdef TIME_CALC

    printf("Total CPU time: %10g sec\n", (double) (tv2.tv_nsec - tv1.tv_nsec)/1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec));
    
#endif

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

#ifdef TIME_CALC
    //ksekina to roloi gia ti metrisi xronou sti GPU
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
#endif
    
    /*****************************CUDA ALLOC*****************************/
    
    CUDAsafeCall(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
    CUDAsafeCall(cudaDeviceSynchronize());
    CUDAsafeCall(cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));
    
    /**************************************************************/
    
    int blockDimX;
    if(imageW <= 32)
	blockDimX = imageW;
    else
	blockDimX = 32;
    
    int gridDimX;
    if((imageW*imageH)/1024 > 0)
	gridDimX = (imageW*imageH)/1024;
    else
	gridDimX = 1;
    
    gridDimX = sqrt(gridDimX);
    if(gridDimX > 65535)
	gridDimX = 65535;
	
    
    dim3 dimBlock(blockDimX, blockDimX);
    dim3 dimGrid(gridDimX, gridDimX);
    
    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
    CUDAsafeCall(cudaPeekAtLastError());
    
    CUDAsafeCall(cudaDeviceSynchronize());
    
    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    CUDAsafeCall(cudaPeekAtLastError());
    
    CUDAsafeCall(cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef TIME_CALC
    //exoume parei tin metrisi gia to xrono tis GPU
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    
#ifdef TIME_CALC
    printf("Total GPU time: %10g sec\n", (double) (tv2.tv_nsec - tv1.tv_nsec)/1000000000.0 + (double) (tv2.tv_sec - tv1.tv_sec));
    
#endif
    
    int err;
    float acc;
    for(acc = maxAccuracy; acc <= minAccuracy; acc *= 10) {
	err = 0;
	for(i = 0; i < (unsigned int) imageW*imageH; i++) {
	    if(acc < ABS(h_OutputCPU[i] - h_OutputGPU[i])) {
		err = 1;
		break;
	    }
	}
	
	if(err == 0) {
#ifndef ACC
	    printf("Max Accuracy: %f\n", acc);
#endif
	    break;
	}
    }
    
    if(err)
#ifndef ACC
	printf("Image is not accurate with filter: %i x %x\n", filter_radius, filter_radius);
#endif
    
    CUDAsafeCall(cudaFree(d_Filter));
    CUDAsafeCall(cudaFree(d_Buffer));
    CUDAsafeCall(cudaFree(d_Input));
    CUDAsafeCall(cudaFree(d_OutputGPU));
    
    free(h_OutputGPU);
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    CUDAsafeCall(cudaDeviceReset());


    return 0;
}
