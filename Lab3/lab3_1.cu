/*
* This sample implements a separable convolution
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define maxAccuracy  	0.00000005
#define minAccuracy	5.00

#define COMMAND_INPUT
#define TIME_CALC
#define ACC

// #define CPU_COMPARISON
#define TILED_CONVOLUTION
#define FILTER_RADIUS 16

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

#ifdef TILED_CONVOLUTION
	__constant__ float const_filter[FILTER_RADIUS*FILTER_RADIUS];

	////////////////////////////////////////////////////////////////////////////////
	// TILED GPU row convolution filter
	////////////////////////////////////////////////////////////////////////////////
	__global__ void tiledconvolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, int row_width, int imageW, int imageH, int filterR, int tile_width, int block, int ratio) {
		extern __shared__ float data_shared[]; //pinakas pou tha fortwthoun ta dedomena sti shared

		int i, j, srcX, srcY;
		int k, d, colX, rowY;

		float sum;

		for(i = 0; i < ratio; i++) {
			for(j = 0; j < ratio; j++) {
				rowY = (i * tile_width)/ratio + threadIdx.y;
				srcY = blockIdx.y * block + rowY + filterR;
				
				colX = (j * tile_width)/ratio + threadIdx.x;
				srcX = blockIdx.x * block + colX + filterR;

				data_shared[rowY * (tile_width + 1) + colX] = d_Src[srcY * row_width + srcX];
				__syncthreads();
			}
		}


		for(i = 0; i < ratio; i++) {
			for(j = 0; j < ratio; j++) {
				rowY = (i * tile_width)/ratio + threadIdx.y;
				srcY = blockIdx.y * block + rowY + filterR;
				
				colX = (j * tile_width)/ratio + threadIdx.x;
				srcX = blockIdx.x * block + colX + filterR;

				sum = 0;

				for(k = -filterR; k <= filterR; k++) {
					d = colX + k;
					//an vriskomaste entos twn oriwn mporoume
					//na xrisimopoihsoume th shared memory
					if(d >= 0 && d < tile_width) {
						sum += data_shared[rowY * (tile_width + 1) + d] * const_filter[filterR - k];
					}
					else {
						sum += d_Src[srcY * row_width + srcX + k] * const_filter[filterR - k];
					}
				}
				d_Dst[srcY * row_width + srcX] = sum;
				__syncthreads();
			}
		}
	}

////////////////////////////////////////////////////////////////////////////////
// TILED GPU column convolution filter
////////////////////////////////////////////////////////////////////////////////
	__global__ void tiledconvolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter, int row_width, int imageW, int imageH, int filterR, int tile_width, int block, int ratio) {
		extern __shared__ float data_shared[]; //pinakas pou tha fortwthoun ta dedomena sti shared

		int i, j, srcX, srcY;
		int k, d, colX, rowY;

		float sum;

		for(i = 0; i < ratio; i++) {
			for(j = 0; j < ratio; j++) {
				rowY = (i * tile_width)/ratio + threadIdx.y;
				srcY = blockIdx.y * block + rowY + filterR;
				
				colX = (j * tile_width)/ratio + threadIdx.x;
				srcX = blockIdx.x * block + colX + filterR;

				data_shared[rowY * (tile_width + 1) + colX] = d_Src[srcY * row_width + srcX];
				__syncthreads();
			}
		}


		for(i = 0; i < ratio; i++) {
			for(j = 0; j < ratio; j++) {
				rowY = (i * tile_width)/ratio + threadIdx.y;
				srcY = blockIdx.y * block + rowY + filterR;
				
				colX = (j * tile_width)/ratio + threadIdx.x;
				srcX = blockIdx.x * block + colX + filterR;

				sum = 0;

				for(k = -filterR; k <= filterR; k++) {
					d = rowY + k;
					//an vriskomaste entos twn oriwn mporoume
					//na xrisimopoihsoume th shared memory
					if(d >= 0 && d < tile_width) {
						sum += data_shared[d * (tile_width + 1) + colX] * const_filter[filterR - k];
					}
					else {
						sum += d_Src[(srcY + k) * row_width + srcX] * const_filter[filterR - k];
					}
				}
				d_Dst[(srcY-filterR) * imageW + srcX - filterR] = sum;
				__syncthreads();
			}
		}
	}

#else

////////////////////////////////////////////////////////////////////////////////
// GPU row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, int row_width,
		       int imageW, int imageH, int filterR) {

    int blockID = (gridDim.x * blockIdx.y) + (gridDim.x * gridDim.y * blockIdx.z) + blockIdx.x;

    int threadID = threadIdx.x + (blockID * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x);

    int x = (threadID % imageH) + filterR;
    int y = (threadID / imageH) + filterR;
    int k, d;

    float sum = 0;

    for (k = -filterR; k <= filterR; k++) {
        d = x + k;
	    sum += d_Src[y * row_width + d] * d_Filter[filterR - k];
    }
    d_Dst[y * row_width + x] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// GPU column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter, int row_width,
				     int imageW, int imageH, int filterR) {

    int blockID = (gridDim.x * blockIdx.y) + (gridDim.x * gridDim.y * blockIdx.z) + blockIdx.x;

    int threadID = threadIdx.x + (blockID * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x);

    int x = (threadID % imageH) + filterR;
    int y = (threadID / imageH) + filterR;
    int k, d;

    float sum = 0;

    for (k = -filterR; k <= filterR; k++) {
        d = y + k;
	    sum += d_Src[d * row_width + x] * d_Filter[filterR - k];

    }
    d_Dst[(y-filterR) * imageH + x - filterR] = sum;
}

#endif

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
    #ifdef TILED_CONVOLUTION
    	if(argc != 4) {
    		printf("Error with command line args.\n");
    		exit(-1);
    	}
    	filter_radius = (unsigned int) atoi(argv[1]);
    	imageW = atoi(argv[2]);
    	int ratio = atoi(argv[3]); //tile to block ratio

    	if(ratio > imageW) {
    		printf("Error, imageW cannot be smaller than ratio.\n");
    		exit(-1);
    	}
    #else
	    if(argc != 3) {
			printf("Error with command line args.\n");
			exit(1);
	    }
	    filter_radius = (unsigned int) atoi(argv[1]);
	    imageW = atoi(argv[2]);
	#endif    

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
#ifndef ACC

    printf("Filter Radius: %i\nFilter Length: %i\n", filter_radius, FILTER_LENGTH);

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);

#endif

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
    *h_OutputGPU,
    *h_InputGPU;

    unsigned int row_width = filter_radius * 2 + imageH;
    unsigned int size = row_width*row_width * sizeof(float);

    h_OutputGPU = (float *) malloc(imageW * imageH * sizeof(float));
    h_InputGPU = (float *) malloc(size);

    CUDAsafeCall(cudaMalloc((void **) &d_Filter, FILTER_LENGTH * sizeof(float)));
    CUDAsafeCall(cudaMalloc((void **) &d_Input, size));
    CUDAsafeCall(cudaMalloc((void **) &d_Buffer, size));
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
    //printf("CPU computation...\n");

#ifdef CPU_COMPARISON
    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
#endif

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas

    /***************PADDING***************/
    memset(h_InputGPU, 0.0, size);
    CUDAsafeCall(cudaMemset(d_Buffer, 0.0, size));

    unsigned int x, y;
    for(i = 0; i < (unsigned int) imageW * imageH; i++) {
		x = filter_radius + i % imageH;
		y = filter_radius + i / imageH;
		h_InputGPU[y * row_width + x] = h_Input[i];
    }

    /*****************************CUDA ALLOC*****************************/

    CUDAsafeCall(cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice));
    CUDAsafeCall(cudaMemcpy(d_Input, h_InputGPU, size, cudaMemcpyHostToDevice));

    /**************************************************************/

#ifdef TILED_CONVOLUTION
    cudaMemcpyToSymbol(const_filter, h_Filter, FILTER_LENGTH * sizeof(float));

    int tile_width, block;
    int blockDimX, gridDimX, shared_mem;

    blockDimX = imageW / ratio;
    if(blockDimX >= 32)
    	blockDimX = 32;

    tile_width = blockDimX * ratio;
    while(tile_width >= 128) {
    	blockDimX /= 2;
    	tile_width = blockDimX * ratio;
    }

    gridDimX = imageW / (blockDimX * ratio);
    block = blockDimX * ratio;

    shared_mem = tile_width * (tile_width + 1) * sizeof(float);

#ifndef ACC
    printf("Tile: %d, %d\ndimBlock: %d, %d\ndimGrid: %d,%d\nShared memory in Bytes: %d\n", tile_width, tile_width, blockDimX, blockDimX, gridDimX, gridDimX, shared_mem);
#endif

    dim3 dimBlock(blockDimX, blockDimX);
    dim3 dimGrid(gridDimX, gridDimX);

#ifdef TIME_CALC
    struct timespec tv1, tv2;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
#endif

    tiledconvolutionRowGPU<<<dimGrid, dimBlock, shared_mem>>>(d_Buffer, d_Input, d_Filter, (int) row_width, imageW, imageH, filter_radius, tile_width, block, ratio);
    CUDAsafeCall(cudaPeekAtLastError());
    CUDAsafeCall(cudaDeviceSynchronize());

    tiledconvolutionRowGPU<<<dimGrid, dimBlock, shared_mem>>>(d_Buffer, d_Input, d_Filter, (int) row_width, imageW, imageH, filter_radius, tile_width, block, ratio);
    CUDAsafeCall(cudaPeekAtLastError());
    CUDAsafeCall(cudaDeviceSynchronize());

#else

    int blockDimX, gridDimX;
    if(imageW <=32)
    	blockDimX = imageW;
    else
    	blockDimX = 32;

    if((imageW*imageH/1024) > 0)
    	gridDimX = imageW*imageH/1024;
    else
    	gridDimX = 1;

    gridDimX = sqrt(gridDimX);
    if(gridDimX > 65535)
    	gridDimX = 65535;

    dim3 dimBlock(blockDimX, blockDimX);
    dim3 dimGrid(gridDimX, gridDimX);

#ifdef TIME_CALC
	//start
    struct timespec tv1, tv2;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
#endif

    convolutionRowGPU<<<dimGrid, dimBlock>>>(d_Buffer, d_Input, d_Filter, (int) row_width, imageW, imageH, filter_radius);
    CUDAsafeCall(cudaPeekAtLastError());
    CUDAsafeCall(cudaDeviceSynchronize());

    convolutionColumnGPU<<<dimGrid, dimBlock>>>(d_OutputGPU, d_Buffer, d_Filter, (int) row_width, imageW, imageH, filter_radius);
    CUDAsafeCall(cudaPeekAtLastError());
    CUDAsafeCall(cudaDeviceSynchronize());
#endif

#ifdef TIME_CALC
    //stamata to roloi
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);
    #ifdef ACC
		printf ("%10g \n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
	#else
		printf ("GPU time: %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));
	#endif
#endif

	CUDAsafeCall(cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW*imageH*sizeof(float), cudaMemcpyDeviceToHost));


#ifdef CPU_COMPARISON
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
    if(err) {
#ifndef ACC
	printf("Image is not accurate with filter: %i x %i\n", filter_radius, filter_radius);
#endif
    }
#endif

    CUDAsafeCall(cudaFree(d_Filter));
    CUDAsafeCall(cudaFree(d_Buffer));
    CUDAsafeCall(cudaFree(d_Input));
    CUDAsafeCall(cudaFree(d_OutputGPU));

    free(h_OutputGPU);
    free(h_InputGPU);
    // free all the allocated memory
    free(h_OutputCPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);

    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    CUDAsafeCall(cudaDeviceReset());

    return 0;
}
