#include "NvInferRuntime.h"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

namespace rfprobe_attention_plugin
{

using namespace nvinfer1;

char constexpr kPLUGIN_NAME[] = "RfProbeEncoderAttentionCore";
char constexpr kPLUGIN_VERSION[] = "1";
int32_t constexpr kNUM_HEADS = 6;
int32_t constexpr kSOFTMAX_THREADS = 256;

cudaDataType_t toCudaType(DataType type) noexcept
{
    switch (type)
    {
    case DataType::kHALF: return CUDA_R_16F;
    case DataType::kFLOAT: return CUDA_R_32F;
    default: return CUDA_R_32F;
    }
}

cublasComputeType_t toComputeType(DataType type) noexcept
{
    switch (type)
    {
    case DataType::kHALF: return CUBLAS_COMPUTE_16F;
    case DataType::kFLOAT: return CUBLAS_COMPUTE_32F;
    default: return CUBLAS_COMPUTE_32F;
    }
}

int64_t elementSize(DataType type) noexcept
{
    switch (type)
    {
    case DataType::kHALF: return 2;
    case DataType::kFLOAT: return 4;
    default: return 4;
    }
}

bool isSameShape(PluginTensorDesc const& a, PluginTensorDesc const& b) noexcept
{
    if (a.type != b.type || a.format != b.format || a.dims.nbDims != b.dims.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < a.dims.nbDims; ++i)
    {
        if (a.dims.d[i] != b.dims.d[i])
        {
            return false;
        }
    }
    return true;
}

template <typename T>
__device__ inline float loadValue(T const* ptr)
{
    return static_cast<float>(*ptr);
}

template <>
__device__ inline float loadValue(__half const* ptr)
{
    return __half2float(*ptr);
}

template <typename T>
__device__ inline T storeValue(float value)
{
    return static_cast<T>(value);
}

template <>
__device__ inline __half storeValue(float value)
{
    return __float2half_rn(value);
}

template <typename InT, typename OutT>
__global__ void packHeadsKernel(
    InT const* input,
    OutT* output,
    int32_t batch,
    int32_t seq,
    int32_t hidden,
    int32_t numHeads,
    int32_t headDim,
    float scale)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = static_cast<int64_t>(batch) * seq * hidden;
    if (idx >= total)
    {
        return;
    }

    int32_t const hiddenIdx = idx % hidden;
    int64_t const tmp = idx / hidden;
    int32_t const seqIdx = tmp % seq;
    int32_t const batchIdx = tmp / seq;
    int32_t const headIdx = hiddenIdx / headDim;
    int32_t const dimIdx = hiddenIdx - headIdx * headDim;
    int64_t const outIdx = ((static_cast<int64_t>(batchIdx) * numHeads + headIdx) * seq + seqIdx) * headDim + dimIdx;
    float const value = loadValue(input + idx) * scale;
    output[outIdx] = storeValue<OutT>(value);
}

template <typename InT, typename OutT>
__global__ void unpackHeadsKernel(
    InT const* input, OutT* output, int32_t batch, int32_t seq, int32_t hidden, int32_t numHeads, int32_t headDim)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = static_cast<int64_t>(batch) * seq * hidden;
    if (idx >= total)
    {
        return;
    }

    int32_t const hiddenIdx = idx % hidden;
    int64_t const tmp = idx / hidden;
    int32_t const seqIdx = tmp % seq;
    int32_t const batchIdx = tmp / seq;
    int32_t const headIdx = hiddenIdx / headDim;
    int32_t const dimIdx = hiddenIdx - headIdx * headDim;
    int64_t const inIdx = ((static_cast<int64_t>(batchIdx) * numHeads + headIdx) * seq + seqIdx) * headDim + dimIdx;
    float const value = loadValue(input + inIdx);
    output[idx] = storeValue<OutT>(value);
}

template <typename T>
__global__ void softmaxRowsKernel(T* scores, int32_t rows, int32_t cols)
{
    int32_t const row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }

    extern __shared__ float shared[];
    T* rowPtr = scores + static_cast<int64_t>(row) * cols;

    float localMax = -1.0e20f;
    for (int32_t col = threadIdx.x; col < cols; col += blockDim.x)
    {
        localMax = fmaxf(localMax, loadValue(rowPtr + col));
    }
    shared[threadIdx.x] = localMax;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float const maxValue = shared[0];

    float localSum = 0.0f;
    for (int32_t col = threadIdx.x; col < cols; col += blockDim.x)
    {
        localSum += expf(loadValue(rowPtr + col) - maxValue);
    }
    shared[threadIdx.x] = localSum;
    __syncthreads();

    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float const invSum = 1.0f / shared[0];

    for (int32_t col = threadIdx.x; col < cols; col += blockDim.x)
    {
        float const value = expf(loadValue(rowPtr + col) - maxValue) * invSum;
        rowPtr[col] = storeValue<T>(value);
    }
}

template <typename InT, typename OutT>
int32_t launchPackKernel(InT const* input,
    OutT* output,
    int32_t batch,
    int32_t seq,
    int32_t hidden,
    int32_t numHeads,
    int32_t headDim,
    float scale,
    cudaStream_t stream) noexcept
{
    int64_t const total = static_cast<int64_t>(batch) * seq * hidden;
    int32_t constexpr block = 256;
    int32_t const grid = static_cast<int32_t>((total + block - 1) / block);
    packHeadsKernel<InT, OutT><<<grid, block, 0, stream>>>(input, output, batch, seq, hidden, numHeads, headDim,
        scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

template <typename InT, typename OutT>
int32_t launchUnpackKernel(InT const* input, OutT* output, int32_t batch, int32_t seq, int32_t hidden,
    int32_t numHeads, int32_t headDim, cudaStream_t stream) noexcept
{
    int64_t const total = static_cast<int64_t>(batch) * seq * hidden;
    int32_t constexpr block = 256;
    int32_t const grid = static_cast<int32_t>((total + block - 1) / block);
    unpackHeadsKernel<InT, OutT><<<grid, block, 0, stream>>>(input, output, batch, seq, hidden, numHeads, headDim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

template <typename T>
int32_t launchSoftmaxKernel(T* scores, int32_t rows, int32_t cols, cudaStream_t stream) noexcept
{
    softmaxRowsKernel<<<rows, kSOFTMAX_THREADS, kSOFTMAX_THREADS * sizeof(float), stream>>>(scores, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

class EncoderAttentionCorePlugin : public IPluginV3,
                                   public IPluginV3OneCore,
                                   public IPluginV3OneBuild,
                                   public IPluginV3OneRuntime
{
public:
    EncoderAttentionCorePlugin() = default;

    EncoderAttentionCorePlugin(EncoderAttentionCorePlugin const& other)
        : mType(other.mType)
        , mComputeType(other.mComputeType)
        , mBatch(other.mBatch)
        , mSeq(other.mSeq)
        , mHidden(other.mHidden)
        , mHeadDim(other.mHeadDim)
        , mBatchHeads(other.mBatchHeads)
        , mPackedBytes(other.mPackedBytes)
        , mScoresBytes(other.mScoresBytes)
        , mWorkspaceBytes(other.mWorkspaceBytes)
        , mFC{}
    {}

    ~EncoderAttentionCorePlugin() override
    {
        destroyState();
    }

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        return new EncoderAttentionCorePlugin(*this);
    }

    char const* getPluginName() const noexcept override
    {
        return kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPLUGIN_VERSION;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t nbInputs, DynamicPluginTensorDesc const*,
        int32_t nbOutputs) noexcept override
    {
        return (nbInputs == 3 && nbOutputs == 1) ? 0 : -1;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 3 || nbOutputs != 1)
        {
            return false;
        }
        auto const& desc = inOut[pos].desc;
        bool const linear = desc.format == TensorFormat::kLINEAR;
        bool const typeOk = desc.type == DataType::kHALF || desc.type == DataType::kFLOAT;
        if (pos == 0)
        {
            return linear && typeOk;
        }
        if (pos <= 2)
        {
            return linear && desc.type == inOut[0].desc.type && desc.format == inOut[0].desc.format;
        }
        return linear && desc.type == inOut[0].desc.type && desc.format == inOut[0].desc.format;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t, DataType const* inputTypes, int32_t) const noexcept override
    {
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t,
        DimsExprs* outputs, int32_t, IExprBuilder&) noexcept override
    {
        if (nbInputs != 3 || inputs[0].nbDims != 3)
        {
            return -1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    size_t getWorkspaceSize(
        DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return static_cast<size_t>(mWorkspaceBytes);
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const*, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 3 || nbOutputs != 1 || in[0].dims.nbDims != 3 || !isSameShape(in[0], in[1])
            || !isSameShape(in[0], in[2]))
        {
            return -1;
        }

        mType = in[0].type;
        mBatch = in[0].dims.d[0];
        mSeq = in[0].dims.d[1];
        mHidden = in[0].dims.d[2];
        if (mBatch <= 0 || mSeq <= 0 || mHidden <= 0 || (mHidden % kNUM_HEADS) != 0)
        {
            return -1;
        }
        mUseFp32Path = false;
        if (mType == DataType::kHALF)
        {
            char const* env = std::getenv("RFPROBE_ENCODER_ATTN_FULL_FP32_PATH");
            if (env != nullptr && env[0] != '\0' && env[0] != '0')
            {
                mUseFp32Path = true;
            }
        }
        mHeadDim = mHidden / kNUM_HEADS;
        mBatchHeads = mBatch * kNUM_HEADS;
        int64_t const elemBytes = mUseFp32Path ? static_cast<int64_t>(sizeof(float)) : elementSize(mType);
        mPackedBytes = static_cast<int64_t>(mBatchHeads) * mSeq * mHeadDim * elemBytes;
        mScoresBytes = static_cast<int64_t>(mBatchHeads) * mSeq * mSeq * elemBytes;
        mWorkspaceBytes = mPackedBytes * 3 + mScoresBytes;
        return createState();
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override
    {
        if (!mReady)
        {
            return -1;
        }

        void* allocatedWorkspace = nullptr;
        if (workspace == nullptr)
        {
            if (cudaMallocAsync(&allocatedWorkspace, static_cast<size_t>(mWorkspaceBytes), stream) != cudaSuccess)
            {
                std::fprintf(stderr, "[RfProbeEncoderAttentionCore] cudaMallocAsync failed bytes=%lld\n",
                    static_cast<long long>(mWorkspaceBytes));
                std::fflush(stderr);
                return -1;
            }
            workspace = allocatedWorkspace;
        }
        auto releaseWorkspace = [&]() {
            if (allocatedWorkspace != nullptr)
            {
                cudaFreeAsync(allocatedWorkspace, stream);
                allocatedWorkspace = nullptr;
            }
        };

        std::byte* base = static_cast<std::byte*>(workspace);
        void* packedQ = base;
        void* packedK = base + mPackedBytes;
        void* packedV = base + 2 * mPackedBytes;
        void* scores = base + 3 * mPackedBytes;
        void* packedO = packedQ;

        int32_t status = 0;
        if (mUseFp32Path)
        {
            status |= launchPackKernel(static_cast<__half const*>(inputs[0]), static_cast<float*>(packedQ), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, mOperandScale, stream);
            status |= launchPackKernel(static_cast<__half const*>(inputs[1]), static_cast<float*>(packedK), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, mOperandScale, stream);
            status |= launchPackKernel(static_cast<__half const*>(inputs[2]), static_cast<float*>(packedV), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, 1.0f, stream);
        }
        else if (mType == DataType::kHALF)
        {
            status |= launchPackKernel(static_cast<__half const*>(inputs[0]), static_cast<__half*>(packedQ), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, mOperandScale, stream);
            status |= launchPackKernel(static_cast<__half const*>(inputs[1]), static_cast<__half*>(packedK), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, mOperandScale, stream);
            status |= launchPackKernel(static_cast<__half const*>(inputs[2]), static_cast<__half*>(packedV), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, 1.0f, stream);
        }
        else
        {
            status |= launchPackKernel(static_cast<float const*>(inputs[0]), static_cast<float*>(packedQ), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, mOperandScale, stream);
            status |= launchPackKernel(static_cast<float const*>(inputs[1]), static_cast<float*>(packedK), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, mOperandScale, stream);
            status |= launchPackKernel(static_cast<float const*>(inputs[2]), static_cast<float*>(packedV), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, 1.0f, stream);
        }
        if (status != 0)
        {
            std::fprintf(stderr, "[RfProbeEncoderAttentionCore] pack failed status=%d\n", status);
            std::fflush(stderr);
            releaseWorkspace();
            return -1;
        }

        if (cublasSetStream(mCublasHandle, stream) != CUBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "[RfProbeEncoderAttentionCore] cublasSetStream failed\n");
            std::fflush(stderr);
            releaseWorkspace();
            return -1;
        }

        cudaDataType_t const ioType = mUseFp32Path ? CUDA_R_32F : toCudaType(mType);
        int64_t const packedStride = static_cast<int64_t>(mSeq) * mHeadDim;
        int64_t const scoresStride = static_cast<int64_t>(mSeq) * mSeq;
        cublasGemmAlgo_t const algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

        cublasStatus_t gemmStatus = cublasGemmStridedBatchedEx(mCublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, mSeq, mSeq,
            mHeadDim, mOne, packedK, ioType, mHeadDim, packedStride, packedQ, ioType, mHeadDim, packedStride,
            mBeta, scores, ioType, mSeq, scoresStride, mBatchHeads, mComputeType, algo);
        if (gemmStatus != CUBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "[RfProbeEncoderAttentionCore] qk gemm failed status=%d\n", int(gemmStatus));
            std::fflush(stderr);
            releaseWorkspace();
            return -1;
        }

        if (mUseFp32Path)
        {
            status = launchSoftmaxKernel(static_cast<float*>(scores), mBatchHeads * mSeq, mSeq, stream);
        }
        else if (mType == DataType::kHALF)
        {
            status = launchSoftmaxKernel(static_cast<__half*>(scores), mBatchHeads * mSeq, mSeq, stream);
        }
        else
        {
            status = launchSoftmaxKernel(static_cast<float*>(scores), mBatchHeads * mSeq, mSeq, stream);
        }
        if (status != 0)
        {
            std::fprintf(stderr, "[RfProbeEncoderAttentionCore] softmax failed status=%d\n", status);
            std::fflush(stderr);
            releaseWorkspace();
            return -1;
        }

        gemmStatus = cublasGemmStridedBatchedEx(mCublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, mHeadDim, mSeq, mSeq, mOne,
            packedV, ioType, mHeadDim, packedStride, scores, ioType, mSeq, scoresStride, mBeta, packedO, ioType,
            mHeadDim, packedStride, mBatchHeads, mComputeType, algo);
        if (gemmStatus != CUBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "[RfProbeEncoderAttentionCore] pv gemm failed status=%d\n", int(gemmStatus));
            std::fflush(stderr);
            releaseWorkspace();
            return -1;
        }

        if (mUseFp32Path)
        {
            status = launchUnpackKernel(static_cast<float const*>(packedO), static_cast<__half*>(outputs[0]), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, stream);
        }
        else if (mType == DataType::kHALF)
        {
            status = launchUnpackKernel(static_cast<__half const*>(packedO), static_cast<__half*>(outputs[0]), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, stream);
        }
        else
        {
            status = launchUnpackKernel(static_cast<float const*>(packedO), static_cast<float*>(outputs[0]), mBatch,
                mSeq, mHidden, kNUM_HEADS, mHeadDim, stream);
        }
        if (status != 0)
        {
            std::fprintf(stderr, "[RfProbeEncoderAttentionCore] unpack failed status=%d\n", status);
            std::fflush(stderr);
            releaseWorkspace();
            return -1;
        }
        releaseWorkspace();
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override
    {
        auto* clonePlugin = new EncoderAttentionCorePlugin(*this);
        if (clonePlugin->mSeq > 0 && clonePlugin->createState() != 0)
        {
            delete clonePlugin;
            return nullptr;
        }
        return clonePlugin;
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFC;
    }

private:
    void destroyState() noexcept
    {
        if (mCublasHandle != nullptr)
        {
            cublasDestroy(mCublasHandle);
            mCublasHandle = nullptr;
        }
        mReady = false;
    }

    int32_t createState() noexcept
    {
        destroyState();
        if (cublasCreate(&mCublasHandle) != CUBLAS_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "[RfProbeEncoderAttentionCore] cublasCreate failed\n");
            std::fflush(stderr);
            return -1;
        }
        mComputeType = mUseFp32Path ? CUBLAS_COMPUTE_32F : toComputeType(mType);

        if (mType == DataType::kHALF)
        {
            char const* env = std::getenv("RFPROBE_ENCODER_ATTN_FP32_COMPUTE");
            if (env != nullptr && env[0] != '\0' && env[0] != '0')
            {
                mComputeType = CUBLAS_COMPUTE_32F;
            }
        }

        float const scale = static_cast<float>(std::sqrt(1.0 / std::sqrt(static_cast<double>(mHeadDim))));
        mOperandScale = scale;
        if (mUseFp32Path || mComputeType == CUBLAS_COMPUTE_32F)
        {
            mOneFloat = 1.0f;
            mZeroFloat = 0.0f;
            mOne = &mOneFloat;
            mBeta = &mZeroFloat;
        }
        else if (mType == DataType::kHALF)
        {
            mOneHalf = __float2half(1.0f);
            mZeroHalf = __float2half(0.0f);
            mOne = &mOneHalf;
            mBeta = &mZeroHalf;
        }
        else
        {
            mOneFloat = 1.0f;
            mZeroFloat = 0.0f;
            mOne = &mOneFloat;
            mBeta = &mZeroFloat;
        }
        char const* debugEnv = std::getenv("RFPROBE_ENCODER_ATTN_DEBUG");
        if (debugEnv != nullptr && debugEnv[0] != '\0' && debugEnv[0] != '0')
        {
            std::fprintf(stderr,
                "[RfProbeEncoderAttentionCore] createState type=%d useFp32Path=%d computeType=%d batch=%d seq=%d hidden=%d workspaceBytes=%lld\n",
                int(mType), int(mUseFp32Path), int(mComputeType), mBatch, mSeq, mHidden,
                static_cast<long long>(mWorkspaceBytes));
            std::fflush(stderr);
        }
        mReady = true;
        return 0;
    }

    DataType mType{DataType::kHALF};
    cublasComputeType_t mComputeType{CUBLAS_COMPUTE_16F};
    int32_t mBatch{0};
    int32_t mSeq{0};
    int32_t mHidden{0};
    int32_t mHeadDim{0};
    int32_t mBatchHeads{0};
    int64_t mPackedBytes{0};
    int64_t mScoresBytes{0};
    int64_t mWorkspaceBytes{0};
    bool mReady{false};
    bool mUseFp32Path{false};

    cublasHandle_t mCublasHandle{nullptr};

    __half mOneHalf{};
    __half mZeroHalf{};
    float mOneFloat{1.0f};
    float mZeroFloat{0.0f};
    float mOperandScale{1.0f};
    void const* mOne{nullptr};
    void const* mBeta{nullptr};

    PluginFieldCollection mFC{};
};

class EncoderAttentionCorePluginCreator : public IPluginCreatorV3One
{
public:
    char const* getPluginName() const noexcept override
    {
        return kPLUGIN_NAME;
    }

    char const* getPluginVersion() const noexcept override
    {
        return kPLUGIN_VERSION;
    }

    PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mFC;
    }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const*, TensorRTPhase) noexcept override
    {
        return new EncoderAttentionCorePlugin();
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

private:
    PluginFieldCollection mFC{};
};

} // namespace rfprobe_attention_plugin

using rfprobe_attention_plugin::EncoderAttentionCorePluginCreator;
REGISTER_TENSORRT_PLUGIN(EncoderAttentionCorePluginCreator);
