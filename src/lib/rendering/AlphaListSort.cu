/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


// #define CUDA_NDEBUG

#include "saiga/vision/torch/CudaHelper.h"
// #include "ATen/cuda/cub.cuh"
#include "saiga/cuda/bitonicSort.h"

#include "AlphaListSort.h"

#ifdef __GNUC__
#    include "cub/cub.cuh"
#endif

inline HD int bfe(int i, int k)
{
    return (i >> k) & 1;
}

struct SortData
{
    float depth;
    int index;
};

HD inline bool operator<(SortData a, SortData b)
{
    return a.depth < b.depth;
}

template <unsigned int shared_memory_amount>
__global__ void SortMultiBitonic(StaticDeviceTensor<int, 1> ticket_counter, StaticDeviceTensor<int, 1> counts,
                                 StaticDeviceTensor<int, 1> scanned_counts, StaticDeviceTensor<int, 2> data)
{
    const int tidInBlock = threadIdx.x;
    const int blockSize  = blockDim.x;

    __shared__ SortData sharedSortingMem[shared_memory_amount];
    __shared__ int ticket;

    if (tidInBlock == 0)
    {
        int* atomic_c_pos = &ticket_counter(0);
        ticket            = atomicAdd(atomic_c_pos, 1);
    }
    __syncthreads();

    // loop over lists
    while (ticket < counts.sizes[0])
    {
        int offset_in_buffer = scanned_counts(ticket);
        int count            = counts(ticket);


        int logN          = ceil(log2(float(count)));
        int sortingAmount = 1 << logN;

        // every thread needs to work
        sortingAmount = max(sortingAmount, blockSize);

        for (int i = tidInBlock; i < shared_memory_amount; i += blockSize)
        {
            if (i < count)
            {
                ((int*)(&sharedSortingMem[i].depth))[0] = data(0, offset_in_buffer + i);
                sharedSortingMem[i].index               = data(1, offset_in_buffer + i);
            }
            else
            {
                const float max_depth     = 1e25;
                sharedSortingMem[i].depth = max_depth;
                sharedSortingMem[i].index = 0;
            }
        }
        __syncthreads();

        // bitonic sort

        int elementsToSortPerThread = ceil((float(sortingAmount) / float(blockSize)));


        for (int stage = 0; stage < logN; stage++)
        {
            for (int i = stage; i >= 0; --i)
            {
                for (int k = 0; k < elementsToSortPerThread; k++)
                {
                    int compareWithElementOffset = 1 << i;
                    int index                    = k * blockSize + tidInBlock;
                    bool up                      = bfe(index, i) ^ bfe(index, stage + 1);
                    int otherIndex               = index ^ compareWithElementOffset;

                    auto getdata = [&](int index) { return sharedSortingMem[index]; };

                    auto setdata = [&](int index, SortData d) { sharedSortingMem[index] = d; };

                    SortData e0 = getdata(index);
                    SortData e1 = getdata(otherIndex);


                    __syncthreads();
                    if (((e0.depth > e1.depth) && !up) || ((e0.depth < e1.depth) && up))
                    {
                        setdata(index, e1);
                        setdata(otherIndex, e0);
                    }
                    __syncthreads();
                }
            }
        }

        if (tidInBlock == 0)
        {
            int* atomic_c_pos = &ticket_counter(0);
            ticket            = atomicAdd(atomic_c_pos, 1);
        }
        __syncthreads();

        for (int i = tidInBlock; i < shared_memory_amount; i += blockSize)
        {
            if (i < count)
            {
                data(0, offset_in_buffer + i) = ((int*)(&sharedSortingMem[i].depth))[0];
                data(1, offset_in_buffer + i) = sharedSortingMem[i].index;
            }
        }
    }
}

template <unsigned int SIZE = 32>
inline __device__ float2 shuffleSwapCompare(float2 x, int mask, int direction)
{
    auto y = Saiga::CUDA::shfl_xor(x, mask, SIZE);
    return x.x < y.x == direction ? y : x;
}


__global__ void SortBitonicWarp(StaticDeviceTensor<int, 1> list_indices, StaticDeviceTensor<int, 1> counts,
                                StaticDeviceTensor<int, 1> scanned_counts, StaticDeviceTensor<int, 2> data)
{
    int tid     = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (warp_id >= list_indices.size(0)) return;

    int list_index = list_indices(warp_id);

    int count  = counts(list_index);
    int offset = scanned_counts(list_index);

    SortData v;

    if (lane_id < count)
    {
        ((int*)(&v.depth))[0] = data(0, offset + lane_id);
        v.index               = data(1, offset + lane_id);
    }
    else
    {
        v.depth = 1e30;
        v.index = 0;
    }


    float2 v_f2;
    ((SortData*)&v_f2)[0] = v;

    v_f2 = CUDA::bitonicWarpSort(v_f2, lane_id);

    v = ((SortData*)&v_f2)[0];

    // v = CUDA::bitonicWarpSort(v, lane_id);

    if (lane_id < count)
    {
        data(0, offset + lane_id) = ((int*)(&v.depth))[0];
        data(1, offset + lane_id) = v.index;
    }
}

template <int shared_memory_amount, int elements_per_thread>
__global__ void SortBitonicInGlobalMemory(StaticDeviceTensor<int, 1> list_indices, StaticDeviceTensor<int, 1> counts,
                                          StaticDeviceTensor<int, 1> scanned_counts, StaticDeviceTensor<int, 2> data)
{
    int block_id = blockIdx.x;
    int lane_id  = threadIdx.x;
    if (block_id >= list_indices.size(0)) return;

    int list_index = list_indices(block_id);


    __shared__ SortData sharedSortingMem[shared_memory_amount];


    int count  = counts(list_index);
    int offset = scanned_counts(list_index);
    SortData all_v[elements_per_thread];

#pragma unroll
    for (int e = 0; e < elements_per_thread; ++e)
    {
        int index = lane_id + e * blockDim.x;

        SortData v;
        if (index < count)
        {
            ((int*)(&v.depth))[0] = data(0, offset + index);
            v.index               = data(1, offset + index);
        }
        else
        {
            v.depth = 1e30;
            v.index = 0;
        }
        sharedSortingMem[index] = v;
        all_v[e]                = v;
    }


    __syncthreads();


    int logN = ceil(log2(float(count)));
    for (int stage = 0; stage < logN; stage++)
    {
        for (int i = stage; i >= 0; --i)
        {
#pragma unroll
            for (int e = 0; e < elements_per_thread; ++e)
            {
                int index                    = lane_id + e * blockDim.x;
                int compareWithElementOffset = 1 << i;
                bool up                      = bfe(index, i) ^ bfe(index, stage + 1);
                int otherIndex               = index ^ compareWithElementOffset;

                auto getdata = [&](int index) { return sharedSortingMem[index]; };

                SortData e1 = getdata(otherIndex);
                auto v      = all_v[e];
                if (((v.depth > e1.depth) && !up) || ((v.depth < e1.depth) && up))
                {
                    all_v[e] = e1;
                }
            }

            __syncthreads();

#pragma unroll
            for (int e = 0; e < elements_per_thread; ++e)
            {
                int index               = lane_id + e * blockDim.x;
                sharedSortingMem[index] = all_v[e];
            }

            __syncthreads();
        }
    }
    __syncthreads();

#pragma unroll
    for (int e = 0; e < elements_per_thread; ++e)
    {
        int index = lane_id + e * blockDim.x;
        auto v    = all_v[e];
        if (index < count)
        {
            data(0, offset + index) = ((int*)(&v.depth))[0];
            data(1, offset + index) = v.index;
        }
    }
}

template <int shared_memory_amount, int elements_per_thread>
__global__ void SortBitonicInBlock(StaticDeviceTensor<int, 1> list_indices, StaticDeviceTensor<int, 1> counts,
                                   StaticDeviceTensor<int, 1> scanned_counts, StaticDeviceTensor<int, 2> data)
{
    int block_id = blockIdx.x;
    int lane_id  = threadIdx.x;
    if (block_id >= list_indices.size(0)) return;

    int list_index = list_indices(block_id);

    __shared__ SortData sharedSortingMem[shared_memory_amount];

    int count = counts(list_index);
    // force disregarding of random elments in too large lists
    count = min(count, shared_memory_amount);

    int offset = scanned_counts(list_index);
    SortData all_v[elements_per_thread];

#pragma unroll
    for (int e = 0; e < elements_per_thread; ++e)
    {
        int index = lane_id + e * blockDim.x;

        SortData v;
        if (index < count)
        {
            ((int*)(&v.depth))[0] = data(0, offset + index);
            v.index               = data(1, offset + index);
        }
        else
        {
            v.depth = 1e30;
            v.index = 0;
        }
        sharedSortingMem[index] = v;
        all_v[e]                = v;
    }


    __syncthreads();


    int logN = ceil(log2(float(count)));
    for (int stage = 0; stage < logN; stage++)
    {
        for (int i = stage; i >= 0; --i)
        {
#pragma unroll
            for (int e = 0; e < elements_per_thread; ++e)
            {
                int index                    = lane_id + e * blockDim.x;
                int compareWithElementOffset = 1 << i;
                bool up                      = bfe(index, i) ^ bfe(index, stage + 1);
                int otherIndex               = index ^ compareWithElementOffset;

                auto getdata = [&](int index) { return sharedSortingMem[index]; };

                SortData e1 = getdata(otherIndex);
                auto v      = all_v[e];
                if (((v.depth > e1.depth) && !up) || ((v.depth < e1.depth) && up))
                {
                    all_v[e] = e1;
                }
            }

            __syncthreads();

#pragma unroll
            for (int e = 0; e < elements_per_thread; ++e)
            {
                int index               = lane_id + e * blockDim.x;
                sharedSortingMem[index] = all_v[e];
            }

            __syncthreads();
        }
    }
    __syncthreads();

#pragma unroll
    for (int e = 0; e < elements_per_thread; ++e)
    {
        int index = lane_id + e * blockDim.x;
        auto v    = all_v[e];
        if (index < count)
        {
            data(0, offset + index) = ((int*)(&v.depth))[0];
            data(1, offset + index) = v.index;
        }
    }
}


template <int shared_memory_amount>
__global__ void SortBitonicLargeBlock(StaticDeviceTensor<int, 1> list_indices, StaticDeviceTensor<int, 1> counts,
                                      StaticDeviceTensor<int, 1> scanned_counts, StaticDeviceTensor<int, 2> data)
{
    int tid       = threadIdx.x + blockIdx.x * blockDim.x;
    int block_id  = blockIdx.x;
    int lane_id   = threadIdx.x;
    int blockSize = blockDim.x;
    if (block_id >= list_indices.size(0)) return;

    int list_index = list_indices(block_id);


    __shared__ SortData sharedSortingMem[shared_memory_amount];


    int count  = counts(list_index);
    int offset = scanned_counts(list_index);
    for (int i = lane_id; i < shared_memory_amount; i += blockSize)
    {
        if (i < count)
        {
            ((int*)(&sharedSortingMem[i].depth))[0] = data(0, offset + i);
            sharedSortingMem[i].index               = data(1, offset + i);
        }
        else
        {
            const float max_depth     = 1e25;
            sharedSortingMem[i].depth = max_depth;
            sharedSortingMem[i].index = 0;
        }
    }
    __syncthreads();


    int logN = ceil(log2(float(count)));
    for (int stage = 0; stage < logN; stage++)
    {
        for (int i = stage; i >= 0; --i)
        {
            for (int index = lane_id; index < shared_memory_amount; index += blockSize)
            {
                int compareWithElementOffset = 1 << i;
                bool up                      = bfe(index, i) ^ bfe(index, stage + 1);
                int otherIndex               = index ^ compareWithElementOffset;

                auto getdata = [&](int index) { return sharedSortingMem[index]; };

                auto setdata = [&](int index, SortData d) { sharedSortingMem[index] = d; };

                SortData e0 = getdata(index);
                SortData e1 = getdata(otherIndex);


                __syncthreads();
                if (((e0.depth > e1.depth) && !up) || ((e0.depth < e1.depth) && up))
                {
                    setdata(index, e1);
                    setdata(otherIndex, e0);
                }
                __syncthreads();
            }
        }
    }
    __syncthreads();

    for (int i = lane_id; i < shared_memory_amount; i += blockSize)
    {
        if (i < count)
        {
            data(0, offset + i) = ((int*)(&sharedSortingMem[i].depth))[0];
            data(1, offset + i) = sharedSortingMem[i].index;
        }
    }
}

void SegmentedSortBitonicHelper(torch::Tensor counts, torch::Tensor scanned_counts, torch::Tensor data,
                                CUDA::CudaTimerSystem* timer)
{
    SAIGA_ASSERT(counts.dim() == 1);
    SAIGA_ASSERT(scanned_counts.dim() == 1);
    SAIGA_ASSERT(data.dim() == 2);
    SAIGA_ASSERT(data.size(0) == 2);



    auto ticket_counter = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    int num_elements    = data.size(1);
    // get max num of elements
    int max_num_lists = counts.max().item<int>();
    if (max_num_lists == 0) return;
    int power_of_two = int(std::ceil(std::log2(float(max_num_lists))));
    // std::cout << TensorInfo(l.per_image_atomic_counters) << " _ " <<
    // TensorInfo(l.per_image_atomic_counters.slice(0,batch,batch+1)) << std::endl; std::cout << "Max num list: " <<
    // max_num_lists << " - po2: " << power_of_two <<std::endl;

    switch (power_of_two)
    {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:  // 32
        {
            ::SortMultiBitonic<32><<<num_elements, 32>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        case 6:  // 64
        {
            ::SortMultiBitonic<64><<<num_elements, 64>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        case 7:  // 128
        {
            ::SortMultiBitonic<128><<<num_elements, 128>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        case 8:  // 256:
        {
            ::SortMultiBitonic<256><<<num_elements, 256>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        case 9:  // 512
        {
            ::SortMultiBitonic<512><<<num_elements, 256>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        case 10:  // 1024
        {
            ::SortMultiBitonic<1024><<<num_elements, 256>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        case 11:  // 2048
        {
            ::SortMultiBitonic<2048><<<num_elements, 256>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        case 12:  // 4096
        {
            ::SortMultiBitonic<4096><<<num_elements, 256>>>(ticket_counter, counts, scanned_counts, data);
            break;
        }
        default:
        {
            SAIGA_ASSERT(false);
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}


__global__ void ComputeSizes(StaticDeviceTensor<int, 1> counts, StaticDeviceTensor<int, 1> size_counter,
                             StaticDeviceTensor<int, 1> local_offset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= counts.size(0)) return;

    int c = counts(tid);

    int o = 0;
    if (c <= 32)
    {
        o = 0;
    }
    else if (c <= 256)
    {
        o = 1;
    }
    else if (c <= 1024)
    {
        o = 2;
    }
    else if (c <= 4096)
    {
        o = 3;
    }
    else
    {
        o = 4;
        // printf("%d", c);
        // CUDA_KERNEL_ASSERT(false);
    }

    int offset        = atomicAdd(&size_counter(o), 1);
    local_offset(tid) = offset;
}


__global__ void ComputeSegmentSizeIndex(StaticDeviceTensor<int, 1> counts, StaticDeviceTensor<int, 1> size_counter,
                                        StaticDeviceTensor<int, 1> local_offset,
                                        StaticDeviceTensor<int, 1> segment_index)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= counts.size(0)) return;

    int c     = counts(tid);
    int local = local_offset(tid);

    int o = 0;
    if (c <= 32)
    {
        o = 0;
    }
    else if (c <= 256)
    {
        o = 1;
    }
    else if (c <= 1024)
    {
        o = 2;
    }
    else if (c <= 4096)
    {
        o = 3;
    }
    else
    {
        o = 4;
        // CUDA_KERNEL_ASSERT(false);
    }

    int global_offset = 0;
    for (int i = 0; i < o; ++i)
    {
        global_offset += size_counter(i);
    }
    global_offset += local;
    segment_index(global_offset) = tid;
}

void SegmentedSortBitonicHelper2(torch::Tensor counts, torch::Tensor scanned_counts, torch::Tensor data,
                                 CUDA::CudaTimerSystem* timer)
{
    const bool verbose = false;

    SAIGA_ASSERT(counts.dim() == 1);
    SAIGA_ASSERT(scanned_counts.dim() == 1);
    SAIGA_ASSERT(data.dim() == 2);
    SAIGA_ASSERT(data.size(0) == 2);

    if (verbose) std::cout << counts << std::endl;

    const int num_buckets = 5;

    auto local_offset = torch::empty_like(counts);
    auto size_counter = torch::zeros({num_buckets}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    {
        SAIGA_OPTIONAL_TIME_MEASURE("ComputeSizes", timer);
        ::ComputeSizes<<<iDivUp(counts.size(0), 128), 128>>>(counts, size_counter, local_offset);
        CUDA_SYNC_CHECK_ERROR();
    }

    auto size_counter_cpu     = size_counter.cpu();
    int* size_counter_cpu_ptr = size_counter_cpu.data_ptr<int>();
    if (verbose)
    {
        PrintTensorInfo(size_counter);
        std::cout << local_offset << std::endl;
        std::cout << size_counter << std::endl;
    }

    auto segment_indices = torch::zeros_like(counts);
    {
        SAIGA_OPTIONAL_TIME_MEASURE("ComputeSegmentSizeIndex", timer);
        ::ComputeSegmentSizeIndex<<<iDivUp(counts.size(0), 128), 128>>>(counts, size_counter, local_offset,
                                                                        segment_indices);
        CUDA_SYNC_CHECK_ERROR();
    }

    if (verbose)
    {
        std::cout << segment_indices << std::endl;
    }


    std::vector<int> size_counter_scan;
    size_counter_scan.push_back(0);
    for (int i = 0; i < num_buckets; ++i)
    {
        size_counter_scan.push_back(size_counter_scan.back() + size_counter_cpu_ptr[i]);
    }


    if (verbose)
    {
        std::cout << "size_counter_scan: ";
        for (auto i : size_counter_scan) std::cout << i << " ";
        std::cout << std::endl;
    }


    if (size_counter_cpu_ptr[0] > 0)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sort32", timer);
        int l = 0;
        if (verbose)
            std::cout << "start warp sorts: " << size_counter_scan[l] << " " << size_counter_scan[l + 1] << std::endl;
        auto list = segment_indices.slice(0, size_counter_scan[l], size_counter_scan[l + 1]);
        SortBitonicWarp<<<iDivUp(size_counter_cpu_ptr[l], 4), 128>>>(list, counts, scanned_counts, data);
    }
    if (size_counter_cpu_ptr[1] > 0)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sort256", timer);
        int l     = 1;
        auto list = segment_indices.slice(0, size_counter_scan[l], size_counter_scan[l + 1]);
        SortBitonicInBlock<256, 1><<<size_counter_cpu_ptr[l], 256>>>(list, counts, scanned_counts, data);
    }
    if (size_counter_cpu_ptr[2] > 0)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sort1024", timer);
        int l     = 2;
        auto list = segment_indices.slice(0, size_counter_scan[l], size_counter_scan[l + 1]);
        SortBitonicInBlock<1024, 4><<<size_counter_cpu_ptr[l], 256>>>(list, counts, scanned_counts, data);
    }

    if (size_counter_cpu_ptr[3] > 0)
    {
        SAIGA_OPTIONAL_TIME_MEASURE("Sort4096", timer);
        int l     = 3;
        auto list = segment_indices.slice(0, size_counter_scan[l], size_counter_scan[l + 1]);
        SortBitonicInBlock<4096, 16><<<size_counter_cpu_ptr[l], 256>>>(list, counts, scanned_counts, data);
    }
    if (size_counter_cpu_ptr[4] > 0)
    {
        int l                = 4;
        auto list            = segment_indices.slice(0, size_counter_scan[l], size_counter_scan[l + 1]);
        auto counts_selected = torch::index_select(counts, 0, list);
        std::cout << "List too long:" << TensorInfo(list) << TensorInfo(counts_selected) << std::endl;
        SortBitonicInBlock<4096, 16><<<size_counter_cpu_ptr[l], 256>>>(list, counts, scanned_counts, data);
    }
    CUDA_SYNC_CHECK_ERROR();
}


void SegmentedSortCubHelper(torch::Tensor counts, torch::Tensor scanned_counts, torch::Tensor& data,
                            CUDA::CudaTimerSystem* timer)
{
#ifdef __GNUC__
    // cub needs the sum after the scan
    scanned_counts = torch::cat({scanned_counts, counts.sum(0, true)}, 0).to(torch::kInt32);


    CHECK(data.is_contiguous());
    float* d_keys_in = (float*)data.data_ptr<int>();
    int* d_values_in = data.data_ptr<int>() + data.stride(0);

    torch::Tensor data_out = torch::zeros_like(data);
    float* d_keys_out      = (float*)data_out.data_ptr<int>();
    int* d_values_out      = data_out.data_ptr<int>() + data_out.stride(0);

    int num_items    = data.size(1);
    int num_segments = counts.size(0);

    int* d_offsets = scanned_counts.data_ptr<int>();


    // Determine temporary device storage requirements
    void* d_temp_storage      = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
                                             d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, 32, 0);
    // Allocate temporary storage
    //    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    auto tmp_tensor = torch::empty({iDivUp((long)temp_storage_bytes, 4L)}, torch::TensorOptions(torch::kCUDA));
    d_temp_storage  = tmp_tensor.data_ptr();
    //  PrintTensorInfo(data_out);

    {
        SAIGA_OPTIONAL_TIME_MEASURE("SortPairs", timer);

        // Run sorting operation
        cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
                                                 d_values_out, num_items, num_segments, d_offsets, d_offsets + 1, 0, 32,
                                                 0);
    }

    // PrintTensorInfo(data);
    // PrintTensorInfo(data_out);
    data.copy_(data_out);
#endif

    //    cudaFree(d_temp_storage);
}
