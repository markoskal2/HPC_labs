#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#define BIN 256
#define THREADS_PER_BLOCK 256
#define TIMES 8 

__global__ void kernel_histogram(int * d_hist_out, unsigned char * d_img_in, int d_img_size);
__global__ void kernel_histogram_equalization(unsigned char * d_img_out, unsigned char * d_img_in, int * d_hist_in, int d_image_size);
void lut_calculation(int *hist, int *lut, int image_size, int nbr_bin);

#define CUDAsafeCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t err, const char *file, int line, bool abort = true) {

    if(err != cudaSuccess) {
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(err), file, line);
		if(abort)
		    exit(err);
    }
}

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    



PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

#endif
