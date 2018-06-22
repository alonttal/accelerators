/* compile with: nvcc -O3 -maxrregcount=32 hw2.cu -o hw2 */
#define WINDOWS

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
#endif  


#include <stdio.h>
#include <assert.h>
#include <string.h>

#define OUT

#define IMG_DIMENSION   32
#define N_IMG_PAIRS     10000
#define NREQUESTS       100000
#define NSTREAMS        64
#define HISTSIZE        256
#define REGS_PER_THREAD 32
#define QUEUE_SLOTS     10
#define THREADBLOCKS    TODO:


#define SQR(a) ((a) * (a))
#define CUDA_CHECK(f) do {                                                                  \
    cudaError_t e = f;                                                                      \
    if (e != cudaSuccess) {                                                                 \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
        exit(1);                                                                            \
    }                                                                                       \
} while (0)

typedef unsigned char uchar;

// We can use the if(n)def to switch easily between win/unix by simply commenting the define code-line
#ifndef WINDOWS
// unix version
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

/* we'll use these to rate limit the request load */
struct rate_limit_t {
    double last_checked;
    double lambda;
    unsigned seed;
};

// We could have done it more generic, i.e. 'one queue fits all' style, but it would require multiple 'cudaHostAlloc's etc.
struct cpu_gpu_queue {
    // We need to pay attention that if we use 'cudaHostAlloc' for the struct's size, the tail and head aren't initialized to 0,
    // and the functions are not defined properly.
    // i.e. We need to do this:
    // alloc -> new struct -> memcpy from struct to alloc
    int tail;
    int head;
    int q[QUEUE_SLOTS * REQUEST_SIZE_BYTES]; // Slot size is different in cpu_gpu and gpu_cpu since request is larger than response

    produce(int* item) {
        if (head < size) {
            CUDA_CHECK( cudaMemcpy(&q[head * REQUEST_SIZE_BYTES], item, REQUEST_SIZE_BYTES, cudaMemcpyHostToHost) );
            head++;
        }
    }
    
    consume(int* item) {
        if (tail < head) {
            CUDA_CHECK( cudaMemcpy(item, &q[tail * REQUEST_SIZE_BYTES], REQUEST_SIZE_BYTES, cudaMemcpyDeviceToDevice) );
            tail++;
            __threadfence();
        }
    }
}

struct gpu_cpu_queue {
    // We need to pay attention that if we use 'cudaHostAlloc' for the struct's size, the tail and head aren't initialized to 0,
    // and the functions are not defined properly.
    // i.e. We need to do this:
    // alloc -> new struct -> memcpy from struct to alloc
    int tail;
    int head;
    int q[QUEUE_SLOTS * RESPONSE_SIZE_BYTES]; // Slot size is different in cpu_gpu and gpu_cpu since request is larger than response

    produce(uchar* item) {
        if (head < size) {
            CUDA_CHECK( cudaMemcpy(&q[head * REQUEST_SIZE_BYTES], item, REQUEST_SIZE_BYTES, cudaMemcpyDeviceToDevice) );
            __threadfence();
            head++;
        }
    }
    
    consume(uchar* item) {
        if (tail < head) {
            CUDA_CHECK( cudaMemcpy(item, &q[tail * REQUEST_SIZE_BYTES], REQUEST_SIZE_BYTES, cudaMemcpyHostToHost) );
            tail++;
        }
    }
}

struct queue_interface {
    (struct cpu_gpu_queue)** cpu_producer_arr;
    (struct cpu_gpu_queue)** cpu_consumer_arr;
    (struct cpu_gpu_queue)** gpu_producer_arr;
    (struct cpu_gpu_queue)** gpu_consumer_arr;
}

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

        printf("%lf\n", (1. / (rate_limit->lambda * 1e-6) * 0.01));
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

struct cmd_line_args {
    int mode;
    int threads_queue_mode;
    double load;
};
struct times {
    double t_start;
    double t_finish;
};
struct request {
    int nreq;
    cudaStream_t stream;
    double distance;
};
enum {PROGRAM_MODE_STREAMS = 0, PROGRAM_MODE_QUEUE};

// read command line parameters
// if parameters are invalid the process might be killed.
struct cmd_line_args read_cmd_params(int argc, char *argv[]) {
    struct cmd_line_args args = {
        -1, /* mode */
        -1, /* threads_queue_mode */
        0   /* load */
    };

    if (argc < 3) print_usage_and_die(argv[0]);

    if (!strcmp(argv[1], "streams")) {
        if (argc != 3) print_usage_and_die(argv[0]);
        args.mode = PROGRAM_MODE_STREAMS;
        args.load = atof(argv[2]);
    } else if (!strcmp(argv[1], "queue")) {
        if (argc != 4) print_usage_and_die(argv[0]);
        args.mode = PROGRAM_MODE_QUEUE;
        args.threads_queue_mode = atoi(argv[2]);
        args.load = atof(argv[3]);
    } else {
        print_usage_and_die(argv[0]);
    }

    return args;
}

// cpu version for calculating the total distance between all images
int h_calc_distance(uchar *images1, uchar *images2, struct times *time) {
    double total_distance;

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

void init_requests(struct request* requests, int nrequests) {
    for (int i = 0; i < nrequests; i++) {
        CUDA_CHECK( cudaStreamCreate(&requests[i].stream) );
        requests[i].distance = 0;
        requests[i].nreq = 0;
    }
}

void destroy_requests(struct request* requests, int nrequests) {
    for (int i = 0; i < nrequests; i++) {
        CUDA_CHECK( cudaStreamDestroy(requests[i].stream) );
    }
}

int stream_calc_distance(uchar *images1, uchar *images2, double load, struct times *time) {
    uchar *gpu_image1, *gpu_image2;
    int *gpu_hist1, *gpu_hist2;
    double *gpu_hist_distance;
    struct rate_limit_t rate_limit;
    struct request requests[NSTREAMS];
    struct request *request;
    double total_distance = 0;
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
    printf("done\n");

    return total_distance;
}

int get_max_threadblocks(int t_per_tb) {
    // int nDevices;
    // cudaGetDeviceCount(&nDevices);
    // printf("%d\n", nDevices);
    int nDevice = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, nDevice);

    int max_tb_by_regs = (prop.regsPerMultiprocessor * prop.multiProcessorCount) / (t_per_tb * REGS_PER_THREAD);
    int max_tb_by_sharedMem = (prop.sharedMemPerMultiprocessor * prop.multiProcessorCount) / (256); // TODO: change to sharedMem size
    int max_tb_by_threads = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / (t_per_tb);
    // TODO: DELETE prinfs
    printf("Device name: %s\n", prop.name);
    printf("Threads per threadblock: %d\n", t_per_tb);
    printf("#SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Shared memory per SM: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Registers per block: %d\n", prop.regsPerBlock);

    return max_tb_by_regs > max_tb_by_sharedMem ? (max_tb_by_regs > max_tb_by_threads ? max_tb_by_regs : max_tb_by_threads) : (max_tb_by_sharedMem > max_tb_by_threads ? max_tb_by_sharedMem : max_tb_by_threads);
}


struct queue_interface create_queue_interface() {
    (struct cpu_gpu_queue)* cpu_producer_arr[THREADBLOCKS]; // i.e. for cpu to pass requests
    (struct cpu_gpu_queue)* cpu_consumer_arr[THREADBLOCKS]; // i.e. for cpu to get responses

    (struct cpu_gpu_queue)* gpu_producer_arr[THREADBLOCKS]; // i.e. for gpu to pass responses
    (struct cpu_gpu_queue)* gpu_consumer_arr[THREADBLOCKS]; // i.e. for gpu to get requests

    struct cpu_gpu_queue tmp_producer;
    tmp_producer.head = 0;
    tmp_producer.tail = 0;

    struct gpu_cpu_queue tmp_consumer;
    tmp_consumer.head = 0;
    tmp_consumer.tail = 0;

    for (int i = 0; i < THREADBLOCKS; i++) {
        CUDA_CHECK( cudaHostAlloc(&cpu_producer_arr[i], sizeof(struct cpu_gpu_queue), 0) );
        CUDA_CHECK( cudaHostAlloc(&cpu_consumer_arr[i], sizeof(struct gpu_cpu_queue), 0) );

        CUDA_CHECK( cudaHostGetDevicePointer(&gpu_producer_arr[i], cpu_consumer_arr[i], 0) );
        CUDA_CHECK( cudaHostGetDevicePointer(&gpu_consumer_arr[i], cpu_producer_arr[i], 0) );

        CUDA_CHECK( cudaMemcpy(cpu_producer_arr[i], &tmp_producer, sizeof(struct cpu_gpu_queue), cudaMemcpyHostToHost) );
        CUDA_CHECK( cudaMemcpy(cpu_consumer_arr[i], &tmp_consumer, sizeof(struct gpu_cpu_queue), cudaMemcpyHostToHost) );
    }

    struct queue_interface _queue_interface;

    _queue_interface.cpu_producer_arr = cpu_producer_arr;
    _queue_interface.cpu_consumer_arr = cpu_consumer_arr;
    _queue_interface.gpu_producer_arr = gpu_producer_arr;
    _queue_interface.gpu_consumer_arr = gpu_consumer_arr;


    // Summary:
    // This function returns the interface for both the cpu and gpu to access the queues.
    // cpu_producer_arr contains the corresponding host pointers for the gpu_consumer_arr device pointers (pointers to queues)
    // cpu_consumer_arr contains the corresponding host pointers for the gpu_producer_arr device pointers (pointers to queues)

    // Usage:
    // Send kernel gpu_consumer_arr
    // Send requests to threadblock i using cpu_producer_arr[i]->produce(host_mem)
    // Each threadblock gets requests using gpu_consumer_arr[blockIdx.x]->consume(device_mem)
    // Response sent using opposite queues (producer <-> consumer)

}

int main(int argc, char *argv[]) {

    struct cmd_line_args args = read_cmd_params(argc, argv);

    uchar *images1; /* we concatenate all images in one huge array */
    uchar *images2;
    CUDA_CHECK( cudaHostAlloc(&images1, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    CUDA_CHECK( cudaHostAlloc(&images2, N_IMG_PAIRS * IMG_DIMENSION * IMG_DIMENSION, 0) );
    load_image_pairs(images1, images2);

    struct times time {0, 0};
    double t_start = 0, t_finish = 0;

    /* using CPU */
    printf("\n=== CPU ===\n");
    double total_distance = h_calc_distance(images1, images2, &time);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (time.t_finish - time.t_start) * 1e+3);


    /* using GPU task-serial.. just to verify the GPU code makes sense */
    // printf("\n=== GPU Task Serial ===\n");
    // do {
    //     double total_distance = d_calc_distance(images1, images2, &time);
    //     printf("average distance between images %f\n", total_distance / NREQUESTS);
    //     printf("throughput = %lf (req/sec)\n", NREQUESTS / (t_finish - t_start) * 1e+3);
    // } while (0);

    /* now for the client-server part */
    printf("\n=== Client-Server ===\n");
    total_distance = 0;
    struct times *vtimes = (struct times *) malloc(NREQUESTS * sizeof(struct times));
    // double *req_t_start = (double *) malloc(NREQUESTS * sizeof(double));
    memset(vtimes, 0, NREQUESTS * sizeof(struct times));

    // double *req_t_end = (double *) malloc(NREQUESTS * sizeof(double));
    // memset(req_t_end, 0, NREQUESTS * sizeof(double));

    // struct rate_limit_t rate_limit;
    // rate_limit_init(&rate_limit, args.load, 0);

    /* TODO allocate / initialize memory, streams, etc... */

    double ti = get_time_msec();
    if (args.mode == PROGRAM_MODE_STREAMS) {
        total_distance = stream_calc_distance(images1, images2, args.load, vtimes);
        // for (int i = 0; i < NREQUESTS; i++) {

        //     /* TODO query (don't block) streams for any completed requests.
        //        update req_t_end of completed requests
        //        update total_distance */

        //     rate_limit_wait(&rate_limit);
        //     req_t_start[i] = get_time_msec();
        //     int img_idx = i % N_IMG_PAIRS;

        //     /* TODO place memcpy's and kernels in a stream */
        // }
        /* TODO now make sure to wait for all streams to finish */
    } else if (args.mode == PROGRAM_MODE_QUEUE) {
        int max_threadblocks = get_max_threadblocks(args.threads_queue_mode); 
        

        // for (int i = 0; i < NREQUESTS; i++) {

        //     /* TODO check producer consumer queue for any responses.
        //        don't block. if no responses are there we'll check again in the next iteration
        //        update req_t_end of completed requests 
        //        update total_distance */

        //     rate_limit_wait(&rate_limit);
        //     int img_idx = i % N_IMG_PAIRS;
        //     req_t_start[i] = get_time_msec();

            /* TODO place memcpy's and kernels in a stream */
        // }
        /* TODO wait until you have responses for all requests */
    } else {
        assert(0);
    }
    double tf = get_time_msec();

    double avg_latency = 0;
    for (int i = 0; i < NREQUESTS; i++) {
        avg_latency += (vtimes[i].t_finish - vtimes[i].t_start);
    }
    avg_latency /= NREQUESTS;

    printf("mode = %s\n", args.mode == PROGRAM_MODE_STREAMS ? "streams" : "queue");
    printf("load = %lf (req/sec)\n", args.load);
    if (args.mode == PROGRAM_MODE_QUEUE) printf("threads = %d\n", args.threads_queue_mode);
    printf("average distance between images %f\n", total_distance / NREQUESTS);
    printf("throughput = %lf (req/sec)\n", NREQUESTS / (tf - ti) * 1e+3);
    printf("average latency = %lf (msec)\n", avg_latency);

    // FREE CUDAHOSTALLOC MEMORY @@@@@@@@@!!!!!!!!!!@@@@@@@@@@@ TODO TODO TODO !!!
    return 0;
}
