/* compile with: nvcc -O3 hw1.cu -o hw1 */
// Itay - Verifying commit privileges
#include <stdio.h>
// linux time lib
// #include <sys/time.h>
// windows time lib
#include <windows.h>

#define IMG_DIMENSION 32
#define N_IMG_PAIRS 10000
#define N_IMG_PAIRS_POW2 10240
#define HIST_SIZE 256
#define SIMPLE 0
#define SHARED 1
#define BATCH_SIMPLE 0
#define BATCH_IMPROVED 1

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

// linux version
// double static inline get_time_msec(void) {
//     struct timeval t;
//     gettimeofday(&t, NULL);
//     return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
// }

// windows version
double static inline get_time_msec(void) {
    SYSTEMTIME time;
    GetSystemTime(&time);
    LONG time_ms = (time.wSecond * 1000) + time.wMilliseconds;
    return time_ms;
}

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

// calling a __global__ function from a __global__ function is only allowed on the compute_35
// architecture or above. So inorder to use it later I'll extract it's functionality to a
// __device__ function.
__device__ void d_image_to_histogram_simple(uchar *image1, OUT int *hist1) {
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
__global__ void image_to_histogram_simple(uchar *image1, OUT int *hist1) {
    d_image_to_histogram_simple(image1, hist1);
}

// calling a __global__ function from a __global__ function is only allowed on the compute_35
// architecture or above. So inorder to use it later I'll extract it's functionality to a
// __device__ function.
__device__ void d_image_to_histogram_shared(uchar *image1, OUT int *hist1) {
    int tid = threadIdx.x;
    // create a shared array which will contain the image.
    __shared__ uchar shared_img[IMG_DIMENSION * IMG_DIMENSION];
    // each thread copies the value from image1 which is located in global memlmry to 
    // shared_img which is located on the shared memory. this requires one global read
    // from each thread.
    shared_img[tid] = image1[tid];
    // create the array of the shared histogram
    __shared__ int shared_hist[HIST_SIZE];
    // initialize the shared histogram values to 0.
    // note that the use of modulo (%) is to save the need of using an "if" branch
    shared_hist[tid % HIST_SIZE] = 0;
    // wait for all threads to finish copying the value from image1 to shared_img and initializing the 
    // shared histogram.
    __syncthreads();
    // after copying the image to the shared memory, compute the pattern which will afterwards
    // be a value in the histogram. This requires each thread another 8 access, but this time 
    // to the shared memory and not the global.
    d_image_to_histogram_simple(shared_img, shared_hist);
    // wait for all threads to finish calculating the pattern and updating the shared histogram.
    __syncthreads();
    // copy the results to the global histogram.
    hist1[tid % HIST_SIZE] = shared_hist[tid % HIST_SIZE];
}
__global__ void image_to_histogram_shared(uchar *image1, OUT int *hist1) {
    d_image_to_histogram_shared(image1, hist1);
}

// an improved version of image_to_histogram_shared function
__device__ void d_image_to_histogram_shared_improved(uchar *image1, OUT int *hist1) {
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
    // wait for all threads to finish copying the value from image1 to shared_img and initializing the 
    // shared histogram.
    __syncthreads();
    // after copying the image to the shared memory, compute the pattern which will afterwards
    // be a value in the histogram. This requires each thread another 8 access, but this time 
    // to the shared memory and not the global.
    uchar pattern = get_binary_pattern(shared_img, i, j);
    // add 1 to the index specified by pattern of the histogram
    atomicAdd(&hist1[pattern], 1);
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

__global__ void image_to_histogram_large_improved(uchar *image1, OUT int *hist1) {
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
    d_image_to_histogram_shared_improved(&image1[k], &hist1[h]);
    __syncthreads();
}

// a function to reduce the distances array. This function works on array of size 256 with 128 threads.
// The param distances should be an array which is shared among the threads.
__device__ void reduce(double *distances, int size) {
    int tid = threadIdx.x;
    int length = size / 2;
    while (length >= 1) {
        if (tid < length) {
            distances[tid] = (double)distances[tid] + (double)distances[tid + length];
        }
        __syncthreads();
        length /= 2;
    }
}
// calling a __global__ function from a __global__ function is only allowed on the compute_35
// architecture or above. So inorder to use it later I'll extract it's functionality to a
// __device__ function.
// this function calculates the distance between to images based on their histograms.
__device__ void d_histogram_distance(int *hist1, int *hist2, OUT double *distance) {
    int tid = threadIdx.x;
    // initialize an array to place the distances in, so later we can reduce it (sum its elements).
    // it is important to use __shared__ here so all the threads in the same thread block share the
    // same array.
    __shared__ double distances [HIST_SIZE];
    // int bid = blockIdx.x;
    // calculate the distance of each position in the histograms, if the sum of both images hist value
    // is zero at the given position then the distance is also zero (this is reason for the use of if-else here). 
    if(hist1[tid] + hist2[tid] > 0) {
        distances[tid] = ((double)SQR(hist1[tid] - hist2[tid])) / (hist1[tid] + hist2[tid]);
        // atomicAdd(distance, add);
    } else {
        distances[tid] = 0;
    }
    // wait for all the threads to complete calculating the distances
    __syncthreads();
    // perform a reduce using half of the threads. We use only have because each thread is summing up
    // 2 elements in each iteration, which means that the sum requires log_2(HIST_SIZE) iterations.
    // we could also use all the threads, and at least half of them would always be idle.
    if (tid < HIST_SIZE / 2) {
        reduce(distances, HIST_SIZE);
    }
    // wait for the reduce to complete and copy the result of the reduce to the OUT pointer.
    __syncthreads();
    // we can do it once using a single thread
    if (tid == 0) {
        *distance = distances[0];
    }
    __syncthreads();
}
__global__ void histogram_distance(int *hist1, int *hist2, OUT double *distance) {
    d_histogram_distance(hist1, hist2, distance);
}

__global__ void histogram_distance_large(int *hist1, int *hist2, OUT double *distances) {
    // int tid = threadIdx.x;
    int bid = blockIdx.x;
    // calculate each pair distance, the histograms of each pair are placed in jumps of HIST_SIZE (256)
    // from one another. Each block-id  will handle a different pair. the result is saved in distance.
    d_histogram_distance(&hist1[HIST_SIZE * bid], &hist2[HIST_SIZE * bid], &distances[bid]);
    __syncthreads();

}

__global__ void hist_reduce_large(double *distances, unsigned int *retirement_count, double OUT *distance) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int offset = blockDim.x * 2; // offset value is 512*2=1024
    // this variable will be used specify the last block passing the fence.
    bool last = false;
    // use the regular reduce this time in chunks of "offset".
    reduce(&distances[bid * offset], offset);
    __syncthreads();
    // after all blocks finished calculation we'll have an array of all images distances,
    // but we still need to reduce this array.
    // first use fence so all the threads in all the thread-blocks share the most up-to-date
    // distances array.
    __threadfence();

    // next, use tickets to find the last thread block and use its 0-thread to sum up the array.
    // the last thread can sum up the array since we know that all the other threads passed the fence.
    // this thread does not have many elements to sum (if i calculated correctly it should have only 
    // (N_IMG_PAIRS_POW2 / offset) = (10240 / 1024) = 10 elements to count)
    if (tid == 0) {
        // inc the ticket for each block finished
        unsigned int ticket = atomicInc(retirement_count, gridDim.x);
        // check if the block is the last block to finish, and if so sum the 10 elements.
        last = (ticket == gridDim.x - 1);
        if (last) {
            *distance = 0;
            for (int i = 0; i < N_IMG_PAIRS_POW2; i += offset) {
                *distance += distances[i];
            }
        }
    }
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

    // first, allocate memory for both images. The allocated memory size should be in the size
    // of one image which is IMG_DIMENSION * IMG_DIMENSION.
    CUDA_CHECK( cudaMalloc(((void**)&gpu_image1), IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)) );
    CUDA_CHECK( cudaMalloc(((void**)&gpu_image2), IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)) );
    // allocate memory for the histograms, each histogram is of size of 256 (0..255).
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist1), HIST_SIZE * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist2), HIST_SIZE * sizeof(int)) );
    // allocate memory for the histogram distance result.
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist_distance), sizeof(double)) );

    double t_start = get_time_msec();

    for (int i = 0; i < N_IMG_PAIRS; i++) {

        // now we need to copy the images from Host to Device. Let us copy the image from the j= i * (IMG_DIMENSION * IMG_DIMENSION)
        // index in each iteration (which is the i'th image).
        CUDA_CHECK( cudaMemcpy(gpu_image1, &images1[i * (IMG_DIMENSION * IMG_DIMENSION)], IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemcpy(gpu_image2, &images2[i * (IMG_DIMENSION * IMG_DIMENSION)], IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar), cudaMemcpyHostToDevice) );
        // intitalize the histograms to all zeros.
        CUDA_CHECK( cudaMemset(gpu_hist1, 0, HIST_SIZE * sizeof(int)) );
        CUDA_CHECK( cudaMemset(gpu_hist2, 0, HIST_SIZE * sizeof(int)) );

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
        
    }     
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_finish = get_time_msec();
    
    // lastly, free all allocated space.
    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    CUDA_CHECK( cudaFree(gpu_hist1) );
    CUDA_CHECK( cudaFree(gpu_hist2) );
    CUDA_CHECK( cudaFree(gpu_hist_distance) );
    double average_distance = total_distance / N_IMG_PAIRS;
    printf("[%s version] GPU average distance: %f\n", (type == SIMPLE ? "SIMPLE" : "SHARED"), average_distance);
    printf("total time %f [msec]\n", t_finish - t_start);
    return average_distance;
}

double gpu_large_average_distance_calculator(uchar *images1, uchar *images2, int type) {
    // the images will be saved here before they are sent to the gpu
    uchar *gpu_image1, *gpu_image2;
    // the histograms calculated on the gpu will be saved here
    int *gpu_hist1, *gpu_hist2;
    // the distances calculated by the gpu in each thread block will be saved here
    double *gpu_hist_distances;
    // the total ditance will be saed here
    double *gpu_hist_distance;
    // the distance calculated on the gpu will be copied back to this cpu variable.
    // this time it is also the total distance calculated (not the average) since
    // we are doing a single iteration (so we calculate everything "at once").
    double cpu_total_distance = 0;
    // this variable will be used to syncronize different thread-blocks.
    unsigned int *retirement_count;

    // first, allocate memory for ALL the images. The allocated memory size should be in the size
    // of all images which is (IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS.
    CUDA_CHECK( cudaMalloc(((void**)&gpu_image1), (IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)) );
    CUDA_CHECK( cudaMalloc(((void**)&gpu_image2), (IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)) );
    // allocate memory for the histograms, each histogram is of size of 256 (0..255).
    // each pair of images has its own histogram, so we need to allocate memory enough for N_IMG_PAIRS
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist1), (HIST_SIZE * N_IMG_PAIRS) * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist2), (HIST_SIZE * N_IMG_PAIRS) * sizeof(int)) );
    // allocate memory for the histogram distances result.
    // this time we are using multiple thread blocks calculation at the same time, each one has its own
    // histogram distance. We need an array big enough to save all these distances. There are total of 
    // N_IMG_PAIRS distances, but we also need a padding of 240 to have a power of 2 (for the reduce).
    // So this should be the size of the the array allocated.
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist_distances), N_IMG_PAIRS_POW2 * sizeof(double)) );
    // allocate memory for the FINAL result which is the sum of all partial sums
    CUDA_CHECK( cudaMalloc(((void**)&gpu_hist_distance), sizeof(double)) );
    // allocate memory for the sync counter.
    CUDA_CHECK( cudaMalloc(((void**)&retirement_count), sizeof(unsigned int)) );
    
    double t_start = get_time_msec();

    // now we need to copy the images from Host to Device.
    // this might take some time because we are copying a pretty "heavy" bundle of data.
    CUDA_CHECK( cudaMemcpy(gpu_image1, images1, ((IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(gpu_image2, images2, ((IMG_DIMENSION * IMG_DIMENSION) * N_IMG_PAIRS * sizeof(uchar)), cudaMemcpyHostToDevice) );
    // intitalize all the array with zeros so the padding will be zeros.
    CUDA_CHECK( cudaMemset(gpu_hist_distances, 0, N_IMG_PAIRS_POW2 * sizeof(double)) );
    // initialize the sync count to 0.
    CUDA_CHECK( cudaMemset(retirement_count, 0, sizeof(int)) );

    // after all those allocations, calculate the images histograms using the "large" version of 
    // image_to_hisogram. This version is doing the calculation on all the images at once, using
    // several threadblocks.
    // run BATCH_SIMPLE algorithm for question 4
    // run BATCH_IMPROVED algorithm for question 6 (bonus)
    if (type == BATCH_SIMPLE) {
        image_to_histogram_large<<<N_IMG_PAIRS, 1024>>>(gpu_image1, gpu_hist1);
        image_to_histogram_large<<<N_IMG_PAIRS, 1024>>>(gpu_image2, gpu_hist2);
    } else if (type == BATCH_IMPROVED) {
        image_to_histogram_large_improved<<<N_IMG_PAIRS, 1024>>>(gpu_image1, gpu_hist1);
        image_to_histogram_large_improved<<<N_IMG_PAIRS, 1024>>>(gpu_image2, gpu_hist2);
    } else {
        printf("error type: %d is not a valid type", type);
    }
    // wait for the histograms calculations to complete.
    CUDA_CHECK( cudaDeviceSynchronize() );
    // calculate the distance of each pair of images using 256 threads (as the size of the histogram - one for each element)
    // for each block. We need N_IMG_PAIRS blocks for N_IMG_PAIRS image-pairs.
    histogram_distance_large<<<N_IMG_PAIRS, 256>>>(gpu_hist1, gpu_hist2, gpu_hist_distances);
    // wait for partial histogram distances calculation to finish.
    CUDA_CHECK( cudaDeviceSynchronize() );
    // reduce the histograms distances. We use 512 threads in each block to sum 1024 elements. we need a total of 
    // ((size of array) / (threads in block)) threadblocks.
    hist_reduce_large<<<(N_IMG_PAIRS_POW2 / 1024), 512>>>(gpu_hist_distances, retirement_count, gpu_hist_distance); 
    // wait for the total distance calculation to finish.
    CUDA_CHECK( cudaDeviceSynchronize() );
    // that's it, we are finally done. From now on everyting is more or less similar to the other "gpu_average_distance_calculator"
    // copy the distance back from the gpu to the cpu.
    CUDA_CHECK( cudaMemcpy(&cpu_total_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost) );
    
    double t_finish = get_time_msec();

    // not to forget freeing the memory!
    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    CUDA_CHECK( cudaFree(gpu_hist1) );
    CUDA_CHECK( cudaFree(gpu_hist2) );
    CUDA_CHECK( cudaFree(gpu_hist_distances) );
    CUDA_CHECK( cudaFree(gpu_hist_distance) );
    CUDA_CHECK( cudaFree(retirement_count) );

    CUDA_CHECK( cudaDeviceSynchronize() );
    double average_distance = cpu_total_distance / N_IMG_PAIRS;
    printf("[%sBATCH version] GPU average distance: %f\n", (type == BATCH_IMPROVED ? "IMPROVED " : ""), average_distance);
    printf("total time %f [msec]\n", t_finish - t_start);

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
    t_start  = get_time_msec();
    for (int i = 0; i < N_IMG_PAIRS; i++) {
        image_to_histogram(&images1[i * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[i * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    t_finish = get_time_msec();
    printf("[CPU version] average distance between images %f\n", total_distance / N_IMG_PAIRS);
    printf("total time %f [msec]\n", t_finish - t_start);




    /* using GPU task-serial */
    printf("\n=== GPU Task Serial ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
        gpu_average_distance_calculator(images1, images2, SIMPLE);
    } while (0);
    


    /* using GPU task-serial + images and histograms in shared memory */
    printf("\n=== GPU Task Serial with shared memory ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
        gpu_average_distance_calculator(images1, images2, SHARED);
    } while (0);
    



    /* using GPU + batching */
    printf("\n=== GPU Batching ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
        gpu_large_average_distance_calculator(images1, images2, BATCH_SIMPLE);
    } while (0);
    



    /* using GPU + improved batching */
    printf("\n=== GPU Batching Improved ===\n");
    do { /* do {} while (0): to keep variables inside this block in their own scope. remove if you prefer otherwise */
        gpu_large_average_distance_calculator(images1, images2, BATCH_IMPROVED);
    } while (0);



    return 0;
}
