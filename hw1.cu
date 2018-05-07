/* compile with: nvcc -O3 hw1.cu -o hw1 */
// Itay - Verifying commit privileges
#include <stdio.h>
// #include <sys/time.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define HIST_SIZE 256
#define SIMPLE 0
#define SHARED 1

typedef unsigned char uchar;
#define OUT

#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

#define SQR(a) ((a) * (a))

// double static inline get_time_msec(void) {
//     struct timeval t;
//     gettimeofday(&t, NULL);
//     return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
// }

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % 256;
        images2[i] = rand() % 256;
    }
}

bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

uchar local_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_in_image_bounds(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_in_image_bounds(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_in_image_bounds(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_in_image_bounds(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_in_image_bounds(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_in_image_bounds(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_in_image_bounds(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_in_image_bounds(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
    return pattern;
}

void image_to_histogram(uchar *image, int *histogram) {
    memset(histogram, 0, sizeof(int) * 256);
    for (int i = 0; i < IMG_DIMENSION; i++) {
        for (int j = 0; j < IMG_DIMENSION; j++) {
            uchar pattern = local_binary_pattern(image, i, j);
            histogram[pattern]++;
        }
    }
}

double histogram_distance(int *h1, int *h2) {
    /* we'll use the chi-square distance */
    double distance = 0;
    for (int i = 0; i < 256; i++) {
        if (h1[i] + h2[i] != 0) {
            distance += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    return distance;
}

/* Your __device__ functions and __global__ kernels here */
/* ... */

/** 
 * I'm implementing atomicAdd for double using atmoicCAS. Note that GPUs with compute capability higher
 * than 6.0 have atomicAdd for double built-in in HW. since my home's pc graphic card only has compute
 * capability of 3.0, it is necessary for me.
*/
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ bool is_valid_position(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}
__device__ uchar get_binary_pattern(uchar *image, int i, int j) {
    uchar center = image[i * IMG_DIMENSION + j];
    uchar pattern = 0;
    if (is_valid_position(i - 1, j - 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j - 1)] >= center) << 7;
    if (is_valid_position(i - 1, j    )) pattern |= (image[(i - 1) * IMG_DIMENSION + (j    )] >= center) << 6;
    if (is_valid_position(i - 1, j + 1)) pattern |= (image[(i - 1) * IMG_DIMENSION + (j + 1)] >= center) << 5;
    if (is_valid_position(i    , j + 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j + 1)] >= center) << 4;
    if (is_valid_position(i + 1, j + 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j + 1)] >= center) << 3;
    if (is_valid_position(i + 1, j    )) pattern |= (image[(i + 1) * IMG_DIMENSION + (j    )] >= center) << 2;
    if (is_valid_position(i + 1, j - 1)) pattern |= (image[(i + 1) * IMG_DIMENSION + (j - 1)] >= center) << 1;
    if (is_valid_position(i    , j - 1)) pattern |= (image[(i    ) * IMG_DIMENSION + (j - 1)] >= center) << 0;
    return pattern;
}

__global__ void image_to_histogram_simple(uchar *image1, OUT int *hist1) {
    int tid = threadIdx.x;
    // int bid = blockIdx.x;
    // if we think about the image as a matrix of dimention (IMG_DIMENSION x IMG_DIMENSION),
    // then i is defining the row and j is defining the column.
    int i = tid / IMG_DIMENSION;
    int j = tid % IMG_DIMENSION;
    // each threads calculated the pattern of his array's element. the pattern is calculated from
    // all (usually 8) neigbouring elements in clockwise order.
    uchar pattern = get_binary_pattern(image1, i, j);
    // atomicaly add 1 to the correct position in the histogram. The position is the pattern calculated.
    // It is important to use atomicAdd here since multiple threads can update the same index concurrently,
    // which may cause racing leading to data corruption.
    atomicAdd(&hist1[pattern], 1);
    // __syncthreads();
}

// calling a __global__ function from a __global__ function is only allowed on the compute_35
// architecture or above. So inorder to use it later I'll extract it's functionality to a
// __device__ function.
__device__ void d_image_to_histogram_shared(uchar *image1, OUT int *hist1) {
    int tid = threadIdx.x;
    // int bid = blockIdx.x;
    // if we think about the image as a matrix of dimention (IMG_DIMENSION x IMG_DIMENSION),
    // then i is defining the row and j is defining the column.
    int i = tid / IMG_DIMENSION;
    int j = tid % IMG_DIMENSION;
    // create a shared array which will contain the image.
    __shared__ uchar shared_img[IMG_DIMENSION * IMG_DIMENSION];
    // each thread copies the value from image1 which is located in global memlmry to 
    // shared_img which is located on the shared memory. this requires one global read
    // from each thread.
    shared_img[tid] = image1[tid];
    // wait for all threads to finish copying the value from image1 to shared_img
    __syncthreads();
    // after copying the image to the shared memory, compute the pattern which will afterwards
    // be a value in the histogram. This requires each thread another 8 access, but this time 
    // to the shared memory and not the global.
    uchar pattern = get_binary_pattern(shared_img, i, j);
    // add 1 to the index specified by pattern of the histogram
    atomicAdd(&hist1[pattern], 1);
}
__global__ void image_to_histogram_shared(uchar *image1, OUT int *hist1) {
    d_image_to_histogram_shared(image1, hist1);
}

__global__ void image_to_histogram_large(uchar *image1, OUT int *hist1) {
    // int tid = threadIdx.x;
    int bid = blockIdx.x;
    // this time we need to calculate the image's number to perform on. k indicates the
    // image number. For example if block-id is 0 then we'll be working on image number 0,
    // if block-id is 1 then we'll be working on image number 1, and so on.. 
    // to find the correct offset we need to multiply each block id by one image size, which
    // is (IMG_DIMENSION * IMG_DIMENSION).
    long k = bid * (IMG_DIMENSION * IMG_DIMENSION);
    // this is the offset in the histogram - jumps of 256 ([0..255], [256..511] and so on..).
    long h = bid * HIST_SIZE;
    // that's it. we have everything we need to know. now the only thing to do is to call the "shared"
    // version of image_to_hisogram, where there it'll use shared memory to calculate the histogram.
    // Note, that each threadblock is calculating a different image (stating in offset of k).
    d_image_to_histogram_shared(&image1[k], &hist1[h]);
    __syncthreads();
}

// calling a __global__ function from a __global__ function is only allowed on the compute_35
// architecture or above. So inorder to use it later I'll extract it's functionality to a
// __device__ function.
// this function calculates the distance between to images based on their histograms.
__device__ void d_histogram_distance(int *hist1, int *hist2, OUT double *distance) {
    int tid = threadIdx.x;
    // int bid = blockIdx.x;
    if(hist1[tid] + hist2[tid] > 0) {
        double add = ((double)SQR(hist1[tid] - hist2[tid])) / (hist1[tid] + hist2[tid]);
        atomicAdd(distance, add);
    }
    __syncthreads();
}
__global__ void histogram_distance(int *hist1, int *hist2, OUT double *distance) {
    d_histogram_distance(hist1, hist2, distance);
}

__global__ void histogram_distance_large(int *hist1, int *hist2, OUT double *distance) {
    // int tid = threadIdx.x;
    int bid = blockIdx.x;
    // calculate each pair distance, the histograms of each pair are placed in jumps of HIST_SIZE (256)
    // from one another. Each block-id  will handle a different pair. the result is saved in distance.
    d_histogram_distance(&hist1[HIST_SIZE * bid], &hist2[HIST_SIZE * bid], distance);
    __syncthreads();
}

double gpu_average_distance_calculator(uchar *images1, uchar *images2, int type) {
    // the images will be saved here before they are sent to the gpu
    uchar *gpu_image1, *gpu_image2;
    // the histograms calculated on the gpu will be saved here
    int *gpu_hist1, *gpu_hist2;
    // the distance calculated by the gpu in each iteration will be saved here
    double *gpu_hist_distance;
    // the distance calculated on the gpu will be copied back to this cpu variable
    double cpu_hist_distance;
    // this is the total distance calculated (not the average)
    double total_distance = 0;

    // t_start = get_time_msec();
    for (int i = 0; i < N_IMG_PAIRS; i++) {
        // first, allocate memory for both images. The allocated memory size should be in the size
        // of one image which is IMG_DIMENSION * IMG_DIMENSION.
        CUDA_CHECK( cudaMalloc(((void**)&gpu_image1), IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)) );
        CUDA_CHECK( cudaMalloc(((void**)&gpu_image2), IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)) );
        // allocate memory for the histograms, each histogram is of size of 256 (0..255).
        CUDA_CHECK( cudaMalloc(((void**)&gpu_hist1), HIST_SIZE * sizeof(int)) );
        CUDA_CHECK( cudaMalloc(((void**)&gpu_hist2), HIST_SIZE * sizeof(int)) );
        // allocate memory for the histogram distance result.
        CUDA_CHECK( cudaMalloc(((void**)&gpu_hist_distance), sizeof(double)) );

        // now we need to copy the images from Host to Device. Let us copy the image from the j= i * (IMG_DIMENSION * IMG_DIMENSION)
        // index in each iteration (which is the i'th image).
        CUDA_CHECK( cudaMemcpy(gpu_image1, &images1[i * (IMG_DIMENSION * IMG_DIMENSION)], IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(gpu_image2, &images2[i * (IMG_DIMENSION * IMG_DIMENSION)], IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar), cudaMemcpyHostToDevice) );
        
        // after all this initialization, calculate the images histograms.
        // type == SIMPLE will use the simple version       (without __shared__)
        // type == SHARED will use the shared image version (__shared__)
        if (type == SIMPLE) {
        image_to_histogram_simple<<<1, 1024>>>(gpu_image1, gpu_hist1);
        image_to_histogram_simple<<<1, 1024>>>(gpu_image2, gpu_hist2);
        } else if (type == SHARED) {
            image_to_histogram_shared<<<1, 1024>>>(gpu_image1, gpu_hist1);
            image_to_histogram_shared<<<1, 1024>>>(gpu_image2, gpu_hist2);
        } else {
            printf("error type: %d is not a valid type", type);
        }
        // wait for the histograms calculations to complete.
        CUDA_CHECK(cudaDeviceSynchronize() );
        
        // calculate the distance using 256 threads (as the size of the histogram - one for each element)
        histogram_distance<<<1, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
        // wait for histogram distance calculation to finish.
        CUDA_CHECK( cudaDeviceSynchronize() );
        // copy the distance back from the gpu to the cpu.
        CUDA_CHECK( cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost) );
        total_distance += cpu_hist_distance;
        
        // lastly, free all allocated space.
        CUDA_CHECK( cudaFree(gpu_image1) );
        CUDA_CHECK( cudaFree(gpu_image2) );
        CUDA_CHECK( cudaFree(gpu_hist1) );
        CUDA_CHECK( cudaFree(gpu_hist2) );
        CUDA_CHECK( cudaFree(gpu_hist_distance) );
    }     
        CUDA_CHECK(cudaDeviceSynchronize());
        double average_distance = total_distance / N_IMG_PAIRS;
        printf("[%s version] GPU average distance: %f\n", (type == SIMPLE ? "SIMPLE" : "SHARED"), average_distance);
        return average_distance;
}

double gpu_large_average_distance_calculator(uchar *images1, uchar *images2) {
    // the images will be saved here before they are sent to the gpu
    uchar *gpu_image1, *gpu_image2;
    // the histograms calculated on the gpu will be saved here
    int *gpu_hist1, *gpu_hist2;
    // the distance calculated by the gpu in each iteration will be saved here
    double *gpu_hist_distance;
    // the distance calculated on the gpu will be copied back to this cpu variable.
    // this time it is also the total distance calculated (not the average) since
    // we are doing a single iteration (so we calculate everything "at once").
    double cpu_total_distance = 0;

    // t_start = get_time_msec();
    // first, allocate memory for ALL the images. The allocated memory size should be in the size
    // of all images which is (IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS.
    CUDA_CHECK( cudaMalloc(((void**)&gpu_image1), (IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)) );
    CUDA_CHECK( cudaMalloc(((void**)&gpu_image2), (IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)) );
    // allocate memory for the histograms, each histogram is of size of 256 (0..255).
    // each pair of images has its own histogram, so we need to allocate memory enough for N_IMG_PAIRS
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist1), (HIST_SIZE * N_IMG_PAIRS) * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist2), (HIST_SIZE * N_IMG_PAIRS) * sizeof(int)) );
    // allocate memory for the histogram distance result.
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist_distance), sizeof(double)) );

    // now we need to copy the images from Host to Device.
    // this might take some time because we are copying a pretty "heavy" bundle of data.
    CUDA_CHECK( cudaMemcpy(gpu_image1, images1, ((IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(gpu_image2, images2, ((IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)), cudaMemcpyHostToDevice) );

    // after all those allocations, calculate the images histograms using the "large" version of 
    // image_to_hisogram. This version is doing the calculation on all the images at once, using
    // several threadblocks.
    image_to_histogram_large<<<N_IMG_PAIRS, 1024>>>(gpu_image1, gpu_hist1);
    image_to_histogram_large<<<N_IMG_PAIRS, 1024>>>(gpu_image2, gpu_hist2);
    // wait for the histograms calculations to complete.
    CUDA_CHECK( cudaDeviceSynchronize() );
    // calculate the distance of each pair of images using 256 threads (as the size of the histogram - one for each element)
    // for each block. We need N_IMG_PAIRS blocks for N_IMG_PAIRS image-pairs.
    histogram_distance_large<<<N_IMG_PAIRS, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
    // that's it, from now on everyting is more or less similar to the other "gpu_average_distance_calculator"
    // wait for histogram distance calculation to finish.
    CUDA_CHECK( cudaDeviceSynchronize() );
    // copy the distance back from the gpu to the cpu.
    CUDA_CHECK( cudaMemcpy(&cpu_total_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost) );
    
    // not to forget freeing the memory!
    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    CUDA_CHECK( cudaFree(gpu_hist1) );
    CUDA_CHECK( cudaFree(gpu_hist2) );
    CUDA_CHECK( cudaFree(gpu_hist_distance) );

    CUDA_CHECK( cudaDeviceSynchronize() );
    double average_distance = cpu_total_distance / N_IMG_PAIRS;
    printf("[LARGE version] GPU average distance: %f\n", average_distance);
    return average_distance;
}

int main() {
    uchar *images1; /* we concatenate all images in one huge array */
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );

    load_image_pairs(images1, images2);
    double t_start, t_finish;
    double total_distance = 0; // Here the course stuff had a mistake, they forgot to initialize total_distance to 0.

    /* using CPU */
    printf("\n=== CPU ===\n");
    int histogram1[256];
    int histogram2[256];
    // t_start  = get_time_msec();
    for (int i = 0; i < N_IMG_PAIRS; i++) {
        image_to_histogram(&images1[i * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[i * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    // t_finish = get_time_msec();
    printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    // printf("total time %f [msec]\n", t_finish - t_start);




    /* using GPU task-serial */
    printf("\n=== GPU Task Serial ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
    gpu_average_distance_calculator(images1, images2, SIMPLE);
        // t_finish = get_time_msec();
        // printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
        // printf("total time %f [msec]\n", t_finish - t_start);
    } while (0);
    


    /* using GPU task-serial + images and histograms in shared memory */
    printf("\n=== GPU Task Serial with shared memory ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
    gpu_average_distance_calculator(images1, images2, SHARED);
        // t_finish = get_time_msec();
        // printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
        // printf("total time %f [msec]\n", t_finish - t_start);
    } while (0);
    /* Your Code Here */
    // printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    // printf("total time %f [msec]\n", t_finish - t_start);
    



    /* using GPU + batching */
    printf("\n=== GPU Batching ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
    gpu_large_average_distance_calculator(images1, images2);
        // t_finish = get_time_msec();
        // printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
        // printf("total time %f [msec]\n", t_finish - t_start);
    } while (0);


    // /* Your Code Here */
    // printf("average distance between images %f\n", total_distance / N_IMG_PAIRS);
    // printf("total time %f [msec]\n", t_finish - t_start);

    return 0;
}
