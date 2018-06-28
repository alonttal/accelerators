/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */
//#define WINDOWS

#ifdef WINDOWS
    #define _CRT_RAND_S
    #include <stdlib.h>
    #include <windows.h>
    #define RANDOM(A) rand()
    #define SLEEP Sleep
#else
    #include <sys/time.h>
    #include <unistd.h>
    #define RANDOM rand_r
    #define SLEEP usleep
    #define UNIX_TIME
#endif  


#include <stdio.h>
#include <assert.h>
#include <string.h>

#define OUT

#define IMG_DIMENSION   32
#define N_IMG_PAIRS     10000
#define NREQUESTS       10000
#define NSTREAMS        64
#define HISTSIZE        256
#define REGS_PER_THREAD 32
#define QUEUE_SIZE      10

#define SQR(a) ((a) * (a))
#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)
#define GPU_GLOBAL_FENCE(a) do { \
    __threadfence_system();      \
    a;                           \
    __threadfence_system();      \
} while(0)

#define SYNCED_WRITE(a) do { \
    __sync_synchronize();    \
    a;                       \
    __sync_synchronize();    \
} while(0)

#ifdef UNIX_TIME
//unix version
double static inline get_time_msec(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}
#else
// windows version
double static inline get_time_msec(void) {
    SYSTEMTIME time;
    GetSystemTime(&time);
    LONG time_ms = (time.wSecond * 1000) + time.wMilliseconds;
    return time_ms;
}
#endif

typedef unsigned char uchar;

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

void rate_limit_init(struct rate_limit_t *rate_limit, double lambda, int seed) {
    rate_limit->lambda = lambda;
    rate_limit->seed = (seed == -1) ? 0 : seed;
    rate_limit->last_checked = 0;
}

int rate_limit_can_send(struct rate_limit_t *rate_limit) {
    if (rate_limit->lambda == 0) return 1;
    double now = get_time_msec() * 1e-3;
    double dt = now - rate_limit->last_checked;
    double p = dt * rate_limit->lambda;
    rate_limit->last_checked = now;
    if (p > 1) p = 1;
    double r = (double)RANDOM(&rate_limit->seed) / RAND_MAX;
    return (p > r);
}

void rate_limit_wait(struct rate_limit_t *rate_limit) {
    while (!rate_limit_can_send(rate_limit)) {
        SLEEP(1. / (rate_limit->lambda * 1e-6) * 0.01);
    }
}

/* we won't load actual files. just fill the images with random bytes */
void load_image_pairs(uchar *images1, uchar *images2) {
    srand(0);
    for (int i = 0; i < N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION; i++) {
        images1[i] = rand() % HISTSIZE;
        images2[i] = rand() % HISTSIZE;
    }
}

__device__ __host__ bool is_in_image_bounds(int i, int j) {
    return (i >= 0) && (i < IMG_DIMENSION) && (j >= 0) && (j < IMG_DIMENSION);
}

__device__ __host__ uchar local_binary_pattern(uchar *image, int i, int j) {
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
    memset(histogram, 0, sizeof(int) * HISTSIZE);
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
    for (int i = 0; i < HISTSIZE; i++) {
        if (h1[i] + h2[i] != 0) {
            distance += ((double)SQR(h1[i] - h2[i])) / (h1[i] + h2[i]);
        }
    }
    // printf("%lf ", distance);

    return distance;
}

__global__ void gpu_image_to_histogram(uchar *image, int *histogram) {
    uchar pattern = local_binary_pattern(image, threadIdx.x / IMG_DIMENSION, threadIdx.x % IMG_DIMENSION);
    atomicAdd(&histogram[pattern], 1);
}

__global__ void gpu_histogram_distance(int *h1, int *h2, double *distance) {
    int length = HISTSIZE;
    int tid = threadIdx.x;
    distance[tid] = 0;
    if (h1[tid] + h2[tid] != 0) {
        distance[tid] = ((double)SQR(h1[tid] - h2[tid])) / (h1[tid] + h2[tid]);
    } else {
        distance[tid] = 0;
    }
    __syncthreads();

    while (length > 1) {
        if (threadIdx.x < length / 2) {
            distance[tid] = distance[tid] + distance[tid + length / 2];
        }
        length /= 2;
        __syncthreads();
    }
}

void print_usage_and_die(char *progname) {
    printf("usage:\n");
    printf("%s streams <load (requests/sec)>\n", progname);
    printf("OR\n");
    printf("%s queue <#threads> <load (requests/sec)>\n", progname);
    exit(1);
}

// enum for specifying current mode (Streams / Queue)
enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};

// struct to save all the command line arguments
// mode                 - the mode of the program (Streams / Queue)
// threads_queue_mode   - is only used for Queue mode and specified the number of threads on a threadblock
// load                 - used to setup the time between each client's request (not working well in windows OS 
//                        only Linux OS due to Windows limitations!!!)
struct cmd_line_args {
    int mode;
    int threads_queue_mode;
    double load;
};

// struct for saving start and end time of job executions
struct times {
    double t_start;
    double t_finish;
};

// this struct is used for the Stream mode
// nreq - the task number that is currently bring handled
// distance- the calculated distance of 2 histograms. This is the taks's result
// stream - cuda streaming channel
struct request {
    int nreq;
    double distance;
    cudaStream_t stream;
};

// producer queue slot (element) holds 2 images to we want to consume
struct producer_queue_slot {
    volatile int req_num;
    volatile int x;
};

// producer queue with the current producer position and current consumer position
struct producer_queue {
    struct producer_queue_slot slots[QUEUE_SIZE];
    volatile int i_producer;
    volatile int i_consumer;
};

// consumer queue slot (element) holds a distance calculation result
struct consumer_queue_slot {
    volatile int req_num;
    volatile double x;
};

// consumer queue with with the current producer position and current consumer position
struct consumer_queue {
    struct consumer_queue_slot slots[QUEUE_SIZE];
    volatile int i_producer;
    volatile int i_consumer;
};

// read command line parameters
// if parameters are invalid the process is killed.
struct cmd_line_args read_cmd_params(int argc, char *argv[]) {
    struct cmd_line_args args = {
        -1, // mode
        -1, // threads_queue_mode
         0  // load
    };

    // if there are not enough arguments - fail
    if (argc < 3) print_usage_and_die(argv[0]);
    // if current mode is Stream, check that there are enough arguments (should be 3),
    // and save user's selected parameters
    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        args.mode = PROGRAM_MODE_STREAMS;
        args.load = atof(argv[2]);
    } 
    // if current mode is Quueue, check that there are enough arguments (should be 4),
    // and save user's selected parameters
    else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        args.mode = PROGRAM_MODE_QUEUE;
        args.threads_queue_mode = atoi(argv[2]);
        args.load = atof(argv[3]);
    } 
    // any other case must be an error - so fail.
    else {
        print_usage_and_die(argv[0]);
    }

    return args;
}

// cpu version for calculating the total distance between all images
// nothing very interesting here.
int h_calc_distance(uchar *images1, uchar *images2, struct times *time) {
    double total_distance = 0;
    int histogram1[HISTSIZE];
    int histogram2[HISTSIZE];

    time->t_start  = get_time_msec();
    for (int i = 0; i < NREQUESTS; i++) {
        int img_idx = i % N_IMG_PAIRS;
        image_to_histogram(&images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram1);
        image_to_histogram(&images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], histogram2);
        total_distance += histogram_distance(histogram1, histogram2);
    }
    time->t_finish = get_time_msec();

    return total_distance;
}

// gpu version for calculating the total distance between all images
// nothing very interesting here.
int d_calc_distance(uchar *images1, uchar *images2, struct times *time) {
    uchar *gpu_image1, *gpu_image2;
    int *gpu_hist1, *gpu_hist2;
    double *gpu_hist_distance;
    double cpu_hist_distance;
    double total_distance;

    cudaMalloc(&gpu_image1, IMG_DIMENSION * IMG_DIMENSION);
    cudaMalloc(&gpu_image2, IMG_DIMENSION * IMG_DIMENSION);
    cudaMalloc(&gpu_hist1, HISTSIZE * sizeof(int));
    cudaMalloc(&gpu_hist2, HISTSIZE * sizeof(int));
    cudaMalloc(&gpu_hist_distance, HISTSIZE * sizeof(double));

    total_distance = 0;
    time->t_start = get_time_msec();
    for (int i = 0; i < NREQUESTS; i++) {
        int img_idx = i % N_IMG_PAIRS;
        cudaMemcpy(gpu_image1, &images1[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_image2, &images2[img_idx * IMG_DIMENSION * IMG_DIMENSION], IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
        cudaMemset(gpu_hist1, 0, HISTSIZE * sizeof(int));
        cudaMemset(gpu_hist2, 0, HISTSIZE * sizeof(int));
        gpu_image_to_histogram<<<1, 1024>>>(gpu_image1, gpu_hist1);
        gpu_image_to_histogram<<<1, 1024>>>(gpu_image2, gpu_hist2);
        gpu_histogram_distance<<<1, HISTSIZE>>>(gpu_hist1, gpu_hist2, gpu_hist_distance);
        cudaMemcpy(&cpu_hist_distance, gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost);
        total_distance += cpu_hist_distance;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    time->t_finish = get_time_msec();

    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    CUDA_CHECK( cudaFree(gpu_hist1) );
    CUDA_CHECK( cudaFree(gpu_hist2) );
    CUDA_CHECK( cudaFree(gpu_hist_distance) );

    return total_distance;
}

// function for initiating array of structs of type request
void init_requests(struct request* requests, int nrequests) {
    // for every request in the array - initiate the cudaStream
    // the 2 other assigns are redundant but it doesn't hurt to set them to zero
    for (int i = 0; i < nrequests; i++) {
        CUDA_CHECK( cudaStreamCreate(&requests[i].stream) );
        requests[i].distance = 0;
        requests[i].nreq = 0;
    }
}
// function for destroying array of structs of type request
void destroy_requests(struct request* requests, int nrequests) {
    // for every request in the array - free the memory of the cudaStream
    for (int i = 0; i < nrequests; i++) {
        CUDA_CHECK( cudaStreamDestroy(requests[i].stream) );
    }
}

// Streams version for calculating the total distance between all images
int stream_calc_distance(uchar *images1, uchar *images2, double load, struct times *time) {
    // array of gpu images where every element belog to a different Stream (used by the gpu)
    uchar *gpu_image1, *gpu_image2;
    // array of gpu histograms where every element belog to a different Stream (used by the gpu)
    int *gpu_hist1, *gpu_hist2;
    // array gpu distance calculation where every element belong to a different Stream (used by the gpu)
    double *gpu_hist_distance;
    // this variable indicates the rate which data will be sent to the gpu
    struct rate_limit_t rate_limit;
    // every stream sent task will recorded here, as well as the stream's result
    struct request requests[NSTREAMS];
    // just a variable to save the current request that is going to be sent to the gpu
    struct request *request;
    // a summer to sum up all calculated distances. at the end, it will hold the desired result
    double total_distance = 0;
    // holds which stream is free to recieve a new task
    int idle;

    // init request struct array which conatains the streams
    init_requests(requests, NSTREAMS);
    // init random variable that handles client requests simulations
    rate_limit_init(&rate_limit, load, 0);
    // init all data structures. the structures should have capacity enough for holding NSTREAMS (64) results.
    CUDA_CHECK( cudaMalloc(&gpu_image1, NSTREAMS * IMG_DIMENSION * IMG_DIMENSION) );
    CUDA_CHECK( cudaMalloc(&gpu_image2, NSTREAMS * IMG_DIMENSION * IMG_DIMENSION) );
    CUDA_CHECK( cudaMalloc(&gpu_hist1, NSTREAMS * HISTSIZE * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&gpu_hist2, NSTREAMS * HISTSIZE * sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&gpu_hist_distance, NSTREAMS * HISTSIZE * sizeof(double)) );

    for (int i = 0; i < NREQUESTS; i++) {
        // find an idle stream. NSTREAM + 1 is not a valid stream number, so this is a good indication for marking that we have 
        // yet to find a stream. i use the term request and stream exchangeably.
        idle = NSTREAMS + 1;
        // brute force loops until you find a free stream
        while (idle > NSTREAMS) { // loop until there is a free stream to use
            for (int j = 0; j < NSTREAMS; j++) {
                request = &requests[j];
                // if the stream status is complete, save it's result, initialize it and save its number so we know it is idle.
                // for the first iteration we will find stream 0 to be idle. It is important to notice that since we initialized
                // the request struct the sum of total_distance does not add to the total sum (since we initialize the request's
                // distance to 0).
                if (cudaStreamQuery(request->stream) == cudaSuccess) {
                    // stop clock ! the end time is saved in the correct position iof the time array. it is known by the request
                    // number that was saved in the request struct (note that it is not mandatory to dave it in the exact position
                    // for correctness, but it was easier for me to implement it that way).
                    time[request->nreq].t_finish = get_time_msec();
                    // it has finished so add its calculated distance to the total sum.
                    total_distance += request->distance;
                    // it is important to initialize the distance so we dont accidentally add it to the total_distance 
                    // later (somehow)!
                    request->distance = 0;
                    // here we save the number of the idle stream/
                    idle = j;
                    // if you find an empty stream, break and use it. no need to continue searching. we will add the distance of others that completed later anyway.
                    break;
                }
            }
        }
        // wait a random amount of time to simulate random client requests arrival.
        rate_limit_wait(&rate_limit);
        // start the clock !
        time[i].t_start = get_time_msec();
        // save the request number so we later know the position in the time array to save the end time.
        request->nreq = i;
        int img_idx = i % N_IMG_PAIRS;
        // before running the kernels in the idle stream, lets make some references so it will be easier to read everything later
        uchar *p_image1 = &images1[img_idx * IMG_DIMENSION * IMG_DIMENSION];
        uchar *p_image2 = &images2[img_idx * IMG_DIMENSION * IMG_DIMENSION];
        uchar *p_gpu_image1 = &gpu_image1[idle * IMG_DIMENSION * IMG_DIMENSION];
        uchar *p_gpu_image2 = &gpu_image2[idle * IMG_DIMENSION * IMG_DIMENSION];
        int *p_gpu_hist1 = &gpu_hist1[idle * HISTSIZE];
        int *p_gpu_hist2 = &gpu_hist2[idle * HISTSIZE];
        double *p_gpu_hist_distance = &gpu_hist_distance[idle * HISTSIZE];
        // now call the kernels. use the idle stream.
        cudaMemcpyAsync(p_gpu_image1, p_image1, IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice, request->stream);
        cudaMemcpyAsync(p_gpu_image2, p_image2, IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice, request->stream);
        cudaMemsetAsync(p_gpu_hist1, 0, HISTSIZE * sizeof(int), request->stream);
        cudaMemsetAsync(p_gpu_hist2, 0, HISTSIZE * sizeof(int), request->stream);
        gpu_image_to_histogram<<<1, 1024, HISTSIZE * sizeof(int), request->stream>>>(p_gpu_image1, p_gpu_hist1);
        gpu_image_to_histogram<<<1, 1024, HISTSIZE * sizeof(int), request->stream>>>(p_gpu_image2, p_gpu_hist2);
        gpu_histogram_distance<<<1, HISTSIZE, HISTSIZE * sizeof(int), request->stream>>>(p_gpu_hist1, p_gpu_hist2, p_gpu_hist_distance);
        cudaMemcpyAsync(&(request->distance), p_gpu_hist_distance, sizeof(double), cudaMemcpyDeviceToHost, request->stream);
        // for the cpu, the kernels are async while for the gpu they are sync for the current stream.
    }
    // wait for all streams to complete
    CUDA_CHECK( cudaDeviceSynchronize() );
    // add their calculated distance to the total_distance. for streams that have already completed, we are just summing zeros
    // because we made sure to zero the distance in the while loop up above.
    for (int j = 0; j < NSTREAMS; j++) {
        request = &requests[j];
        time[request->nreq].t_finish = get_time_msec();
        total_distance += request->distance;
    }

    // Don't forget to free !
    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );
    CUDA_CHECK( cudaFree(gpu_hist1) );
    CUDA_CHECK( cudaFree(gpu_hist2) );
    CUDA_CHECK( cudaFree(gpu_hist_distance) );
    destroy_requests(requests, NSTREAMS);

    return total_distance;
}

// get the number of parallel thread blocks that can cocurrently run on the machine.
// this function considers 3 gpu limitations:
// 1. registers limitation - we know that every thread is going to use exactly 32 registers (as specified in the 
// compilation command). Then, we need to take the number registers in each SM and divide it by the number of register
// each thread uses to get the number of threads that can run on that SM. Divide the result by the number of threads
// defined in the command line arguments to get the number of threadblocks which can run on that SM. lastly, multiply
// by the number of SM in the device to get the total number of threadblocks.
// 2. Shared memory - we are going to use shared memory in this section to share the memory of the histogram and distances
// calculations. Dues we want to make sure we are not acceding the total usage of shared memory. we take the total shared
// memory a SM possess and divide it by the shared memory we are going to use (in bytes). afterwards we multiply it by the
// number of SMs.
// 3. Threads boundary - we might be bounded by the nubmer of threads each SM can run. We take the total number of threads
// an SM can run and divide it by the number of threads in each threadblock to get the number of threadblocks that can run simultaneously on an SM. after that multiply by the number of SM to get total number of threadblocks that can run together.
int get_max_threadblocks(int t_per_tb) {
    int nDevice = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, nDevice);

    // total threadblocks number might be blocked by the number of register each thread uses.
    int max_tb_by_regs = (prop.regsPerMultiprocessor / (t_per_tb * REGS_PER_THREAD)) * prop.multiProcessorCount;
    // total threadblocks number might be blocked by the number of shared memory each threadblock uses.
    int max_tb_by_sharedMem = (prop.sharedMemPerMultiprocessor / (2 * HISTSIZE * sizeof(int) + HISTSIZE * sizeof(double))) *prop.multiProcessorCount;
    // total threadblocks number might be blocked by the number of concurrent running threads. 
    int max_tb_by_threads = (prop.maxThreadsPerMultiProcessor / t_per_tb) * prop.multiProcessorCount;

    // the max total threads is the minimum between all theses 3 arguments (min-max condition)
    return max_tb_by_regs < max_tb_by_sharedMem ? (max_tb_by_regs < max_tb_by_threads ? max_tb_by_regs : max_tb_by_threads) : (max_tb_by_sharedMem < max_tb_by_threads ? max_tb_by_sharedMem : max_tb_by_threads);
}

// see if there are pending tasks that need to be taken care
__device__ void has_pending_tasks(volatile struct producer_queue *queue, bool *res) {
    // pretending that I am the consumer - looking at the queue
    // if the producer position is bigger than my position it means that he has placed a task for me
    *res = queue->i_producer > queue->i_consumer;
}

// init histogram with zeros
__device__ void initHistogram(int *hist) {
    int tid = threadIdx.x;
    int nThreads = blockDim.x;

    for (int i = tid; i < HISTSIZE; i += nThreads) {
        hist[i] = 0;
    }
}

// convert an image to hsitogram (# of threads can be any #)
__device__ void im2hist(uchar *image, int *histogram) {
    int tid = threadIdx.x;
    int nThreads = blockDim.x;

    initHistogram(histogram);
    __syncthreads();
    
    for(int i = tid ; i < IMG_DIMENSION * IMG_DIMENSION; i += nThreads) {
        uchar pattern = local_binary_pattern(image, i / IMG_DIMENSION, i % IMG_DIMENSION);
        atomicAdd(&histogram[pattern], 1);
    }
}

// calculate distance between 2 histograms (# of threads can be any #)
__device__ void hists2dist(int *h1, int *h2, double *distance) {
    int tid = threadIdx.x;
    int nThreads = blockDim.x;

    for (int i = tid; i < HISTSIZE ; i += nThreads) {
        double sum = h1[i] + h2[i];
        distance[i] = (sum != 0) ? ((double)SQR(h1[i] - h2[i])) / sum : 0;
    }
    
    int half_length = HISTSIZE / 2;
    __syncthreads();

	while (half_length >= 1) {
		for (int i = tid; i < half_length; i += nThreads) {
            distance[i] = distance[i] + distance[i + half_length];
        }
		half_length /= 2;
		__syncthreads();
    }
}

__global__ void gpu_consume(uchar* image1, uchar* image2, 
                            volatile int *done,
                            volatile struct producer_queue *pc_queue,
                            volatile struct consumer_queue *cp_queue) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    //histograms must be shared between threads
    __shared__ int histogram1[HISTSIZE]; 
    __shared__ int histogram2[HISTSIZE];
    __shared__ double distance[HISTSIZE];

    while (true) {

        if (tid == 0) {
            while (pc_queue[bid].i_producer <= pc_queue[bid].i_consumer && !(*done)); // wait for new tasks
        }

        __syncthreads();

        if(pc_queue[bid].i_producer <= pc_queue[bid].i_consumer && (*done)) {
            break; // end
        }

        int img_idx = pc_queue[bid].slots[(pc_queue[bid].i_consumer) % QUEUE_SIZE].x;
        int req_num = pc_queue[bid].slots[(pc_queue[bid].i_consumer) % QUEUE_SIZE].req_num;

        __syncthreads();

        if (tid == 0) {
            GPU_GLOBAL_FENCE(pc_queue[bid].i_consumer++);
        } 
        
        // calculate the images histograms
        im2hist(&(image1[img_idx]), histogram1);
        im2hist(&(image2[img_idx]), histogram2);
        // wait for all thread to finish calculations
        __syncthreads();
        // calculate histograms distance
        hists2dist(histogram1, histogram2, distance);
        // wait for all thread to finish calculations
        __syncthreads();
        // let only the 0 thread update the producer
        if (tid == 0) {
            while (cp_queue[bid].i_consumer - cp_queue[bid].i_producer >= QUEUE_SIZE); // wait for space in queue
            
            GPU_GLOBAL_FENCE(cp_queue[bid].slots[(cp_queue[bid].i_consumer) % QUEUE_SIZE].req_num = req_num);
            GPU_GLOBAL_FENCE(cp_queue[bid].slots[(cp_queue[bid].i_consumer) % QUEUE_SIZE].x = distance[0]);
            GPU_GLOBAL_FENCE(cp_queue[bid].i_consumer++);
        }
        __syncthreads();
    }
}

// return the sum of distances computed by a consumer
double collect_completed_tasks_results(volatile struct consumer_queue *queue, struct times *time) {
    double distance = 0;
    // this value may change, so before we start lets save it
    volatile int last_completed_task = queue->i_consumer;
    // for every completed task copy the result and assign the end time
    for(int i = queue->i_producer; i < last_completed_task; i++) {
        distance += queue->slots[i % QUEUE_SIZE].x;
        time[queue->slots[i % QUEUE_SIZE].req_num].t_finish = get_time_msec();
    }
    // increase the producer position by the number of the completed tasks gathered
    SYNCED_WRITE(queue->i_producer = last_completed_task);

    return distance;
}

void send_task(int task_num, int img_idx, volatile struct producer_queue *queue, struct times *time) {
    // save the start time of the request
    time[task_num].t_start = get_time_msec();
    // calculate the slot in the producer queue the task should be placed in
    int slot_index = (queue->i_producer) % QUEUE_SIZE;
    // save the task number in its coresponding slot_index so we later know the index in time array to save the finish_time
    SYNCED_WRITE(queue->slots[slot_index].req_num = task_num);
    // copy the job to the comsumer
    SYNCED_WRITE(queue->slots[slot_index].x = img_idx);
    // now the consumer has all the data it needs to compute the distance between the 2 images. let him know there is a new task
    // by increasing the i_producer variable.
    SYNCED_WRITE(queue->i_producer = queue->i_producer + 1);
}

inline bool can_send_tasks(volatile struct producer_queue *pc_queue) {
    // if the producer send-receive window is less than QUEUE_SIZE it means we will not override any other slot - i.e. we
    // can send a new task to the consumer
    return pc_queue->i_producer - pc_queue->i_consumer < QUEUE_SIZE;
}

inline bool has_completed_tasks(volatile struct consumer_queue *cp_queue) {
    // if the position of the consumer is larger then the postion of the producer, then there are completed tasks
    return cp_queue->i_producer < cp_queue->i_consumer;
}

// producer-consumer version for calculating the total distance between all images
int producer_consumer_calc_distance(uchar *images1, uchar *images2, double load, struct times *time, int nConsumers, int t_per_tb) {
    uchar *gpu_image1, *gpu_image2;
    struct rate_limit_t rate_limit; // this variable indicates the rate which data will be sent to the gpu
    volatile struct producer_queue *pc_queue, *gpu_pc_queue; // producer-consumer queue
    volatile struct consumer_queue *cp_queue, *gpu_cp_queue; // consumer-producer queue
    volatile int *done, *gpu_done; // flag that will indicate that there are no tasks left
    double total_distance = 0;
    int i_next_consumer = 0; // will save the next consumer to recieve a job
    int left_requests = NREQUESTS; // will save the # of jobs left to handle

    // init random variable that handles client requests simulations
    rate_limit_init(&rate_limit, load, 0);
    // init producer-consumer & consumer-producer queue
    // share pointers of the queues between cpu and gpu
    CUDA_CHECK( cudaHostAlloc(&done, sizeof(int), 0) );
    CUDA_CHECK( cudaHostAlloc(&pc_queue, nConsumers * sizeof(struct producer_queue), 0) );
    CUDA_CHECK( cudaHostAlloc(&cp_queue, nConsumers * sizeof(struct consumer_queue), 0) );
    CUDA_CHECK( cudaHostGetDevicePointer((void**)&gpu_done, (void*)done, 0) );
    CUDA_CHECK( cudaHostGetDevicePointer((void**)&gpu_pc_queue, (void*)pc_queue, 0) );
    CUDA_CHECK( cudaHostGetDevicePointer((void**)&gpu_cp_queue, (void*)cp_queue, 0) );
    // init gpu images
    CUDA_CHECK( cudaMalloc(&gpu_image2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)) );
    CUDA_CHECK( cudaMalloc(&gpu_image1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION * sizeof(uchar)) );
    cudaMemcpy(gpu_image1, images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_image2, images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, cudaMemcpyHostToDevice);

    for (int i = 0; i < nConsumers ; i++) {
        pc_queue[i].i_producer = 0;
        pc_queue[i].i_consumer = 0;
        cp_queue[i].i_producer = 0;
        cp_queue[i].i_consumer = 0;
    }

    SYNCED_WRITE(*done = 0); // set the flag to done = false, so we indicate that there is still some work to be done
    CUDA_CHECK( cudaDeviceSynchronize() ); // sync everything with the gpu before starting

    gpu_consume<<<nConsumers, t_per_tb>>>(gpu_image1, gpu_image2, gpu_done, gpu_pc_queue, gpu_cp_queue);

    while (left_requests > 0) {
        if (has_completed_tasks( &(cp_queue[i_next_consumer]) )) { // collect completed tasks
            total_distance += collect_completed_tasks_results(&(cp_queue[i_next_consumer]), time);
        }

        rate_limit_wait(&rate_limit); // wait

        if (can_send_tasks( &(pc_queue[i_next_consumer]) )) { // send new tasks
            int req_num = NREQUESTS - left_requests;
            int img_idx = (req_num % N_IMG_PAIRS) * IMG_DIMENSION * IMG_DIMENSION;
            // send task!
            // printf("sending task: %d to %d. image index is: %d\n", req_num, i_next_consumer, img_idx);
            send_task(req_num, img_idx, &(pc_queue[i_next_consumer]), time);
            --left_requests;
        }
        i_next_consumer = (i_next_consumer + 1) % nConsumers; // move to next consumer
    }

    SYNCED_WRITE(*done = 1); // we can tell all consumers that there are no more tasks left

    // collect remaining results
    for (int i = 0 ; i < nConsumers; i++) {
        do {
            total_distance += collect_completed_tasks_results(&(cp_queue[i]), time);
        } while(cp_queue[i].i_consumer < pc_queue[i].i_producer);
        total_distance += collect_completed_tasks_results(&(cp_queue[i]), time);
    }

    // free!!!
    CUDA_CHECK( cudaFreeHost((void*)done) ); 
    CUDA_CHECK( cudaFreeHost((void*)pc_queue) );
    CUDA_CHECK( cudaFreeHost((void*)cp_queue) );
    CUDA_CHECK( cudaFree(gpu_image1) );
    CUDA_CHECK( cudaFree(gpu_image2) );

    return total_distance;
}

int main(int argc, char *argv[]) {
    double total_distance = 0;
    struct cmd_line_args args = read_cmd_params(argc, argv);

    uchar *images1; /* we concatenate all images in one huge array */
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    load_image_pairs(images1, images2);

    struct times time = {0, 0};
    double t_start = 0, t_finish = 0;

    /* using CPU */
    printf("\n=== CPU ===\n");
    total_distance = h_calc_distance(images1, images2, &time);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (time.t_finish - time.t_start) * 1e+3);


    /* using GPU task-serial.. just to verify the GPU code makes sense */
    printf("\n=== GPU Task Serial ===\n");
    do {
        double total_distance = d_calc_distance(images1, images2, &time);
        printf("average distance between images %f\n", total_distance / NREQUESTS);
        printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);
    } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    total_distance = 0;
    struct times *vtimes = (struct times *) malloc(NREQUESTS * sizeof(struct times));
    memset(vtimes, 0, NREQUESTS * sizeof(struct times));

    double ti = get_time_msec();
    if (args.mode == PROGRAM_MODE_STREAMS) {

        total_distance = stream_calc_distance(images1, images2, args.load, vtimes);

    } else if (args.mode == PROGRAM_MODE_QUEUE) {
        int max_threadblocks = get_max_threadblocks(args.threads_queue_mode);
        // printf("max threadblocks: %d\n", max_threadblocks); 
        total_distance = producer_consumer_calc_distance(images1, images2, args.load, vtimes, max_threadblocks, args.threads_queue_mode);
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (vtimes[i].t_finish - vtimes[i].t_start);
        // printf("%lf %lf\n", vtimes[i].t_start,vtimes[i].t_finish);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", args.mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", args.load);
    if (args.mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", args.threads_queue_mode);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    free(vtimes);
    CUDA_CHECK( cudaFreeHost(images1) );
    CUDA_CHECK( cudaFreeHost(images2) );

    return 0;
}
