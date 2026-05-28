#include "NvInferRuntime.h"

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>

namespace
{

using namespace nvinfer1;

char constexpr kPLUGIN_NAME[] = "RfProbeProjectionMatmul";
char constexpr kPLUGIN_VERSION[] = "1";
size_t constexpr kWORKSPACE_SIZE = 1 << 20;

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

size_t getLeadingVolume(Dims const& dims) noexcept
{
    size_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims - 1; ++i)
    {
        volume *= static_cast<size_t>(dims.d[i]);
    }
    return volume;
}

class ProjectionMatmulPlugin : public IPluginV3,
                               public IPluginV3OneCore,
                               public IPluginV3OneBuild,
                               public IPluginV3OneRuntime
{
public:
    ProjectionMatmulPlugin() = default;

    ProjectionMatmulPlugin(ProjectionMatmulPlugin const& other)
        : mType(other.mType)
        , mComputeType(other.mComputeType)
        , mM(other.mM)
        , mN(other.mN)
        , mK(other.mK)
        , mFC{}
    {}

    ~ProjectionMatmulPlugin() override
    {
        destroyLtState();
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
        return new ProjectionMatmulPlugin(*this);
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
        return (nbInputs == 2 && nbOutputs == 1) ? 0 : -1;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 2 || nbOutputs != 1)
        {
            return false;
        }

        auto const& desc = inOut[pos].desc;
        bool const linear = desc.format == TensorFormat::kLINEAR;
        bool const inputTypeOk = desc.type == DataType::kFLOAT || desc.type == DataType::kHALF;

        if (pos == 0)
        {
            return linear && inputTypeOk;
        }
        if (pos == 1)
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
        if (nbInputs != 2 || inputs[0].nbDims < 2 || inputs[1].nbDims != 2)
        {
            return -1;
        }
        outputs[0].nbDims = inputs[0].nbDims;
        for (int32_t i = 0; i < inputs[0].nbDims - 1; ++i)
        {
            outputs[0].d[i] = inputs[0].d[i];
        }
        outputs[0].d[inputs[0].nbDims - 1] = inputs[1].d[1];
        return 0;
    }

    size_t getWorkspaceSize(
        DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return kWORKSPACE_SIZE;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const*, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 2 || nbOutputs != 1 || in[0].dims.nbDims < 2 || in[1].dims.nbDims != 2)
        {
            return -1;
        }

        mType = in[0].type;
        mComputeType = toComputeType(mType);
        mM = static_cast<int64_t>(getLeadingVolume(in[0].dims));
        mK = static_cast<int64_t>(in[0].dims.d[in[0].dims.nbDims - 1]);
        mN = static_cast<int64_t>(in[1].dims.d[1]);
        if (mM <= 0 || mN <= 0 || mK <= 0)
        {
            return -1;
        }

        return createLtDescriptors();
    }

    int32_t enqueue(PluginTensorDesc const*, PluginTensorDesc const*, void const* const* inputs, void* const* outputs,
        void* workspace, cudaStream_t stream) noexcept override
    {
        if (!mAlgoAvailable)
        {
            return -1;
        }

        if (mType == DataType::kHALF)
        {
            __half alpha = __float2half(1.0f);
            __half beta = __float2half(0.0f);
            cublasStatus_t const status = cublasLtMatmul(mLtHandle, mOperationDesc, &alpha, inputs[1], mADesc, inputs[0],
                mBDesc, &beta, outputs[0], mCDesc, outputs[0], mCDesc, &mAlgo,
                workspace, kWORKSPACE_SIZE, stream);
            return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
        }

        float alpha = 1.0f;
        float beta = 0.0f;
        cublasStatus_t const status = cublasLtMatmul(mLtHandle, mOperationDesc, &alpha, inputs[1], mADesc, inputs[0],
            mBDesc, &beta, outputs[0], mCDesc, outputs[0], mCDesc, &mAlgo,
            workspace, kWORKSPACE_SIZE, stream);
        return status == CUBLAS_STATUS_SUCCESS ? 0 : -1;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override
    {
        auto* clonePlugin = new ProjectionMatmulPlugin(*this);
        if (clonePlugin->mM > 0 && clonePlugin->createLtDescriptors() != 0)
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
    void destroyLtState() noexcept
    {
        if (mADesc != nullptr)
        {
            cublasLtMatrixLayoutDestroy(mADesc);
            mADesc = nullptr;
        }
        if (mBDesc != nullptr)
        {
            cublasLtMatrixLayoutDestroy(mBDesc);
            mBDesc = nullptr;
        }
        if (mCDesc != nullptr)
        {
            cublasLtMatrixLayoutDestroy(mCDesc);
            mCDesc = nullptr;
        }
        if (mOperationDesc != nullptr)
        {
            cublasLtMatmulDescDestroy(mOperationDesc);
            mOperationDesc = nullptr;
        }
        if (mLtHandle != nullptr)
        {
            cublasLtDestroy(mLtHandle);
            mLtHandle = nullptr;
        }
        mAlgoAvailable = false;
    }

    int32_t createLtDescriptors() noexcept
    {
        destroyLtState();

        if (cublasLtCreate(&mLtHandle) != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }

        cudaDataType_t const ioType = toCudaType(mType);
        cudaDataType_t const scaleType = (mType == DataType::kHALF) ? CUDA_R_16F : CUDA_R_32F;
        if (cublasLtMatmulDescCreate(&mOperationDesc, mComputeType, scaleType) != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }

        cublasOperation_t opN = CUBLAS_OP_N;
        if (cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN))
            != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }
        if (cublasLtMatmulDescSetAttribute(mOperationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN))
            != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }

        // Match TensorRT fcPlugin's column-major GEMM convention:
        // weights (K x N row-major) -> A as (N x K) column-major, ld = N
        // activations (M x K row-major) -> B as (K x M) column-major, ld = K
        // output (M x N row-major) -> C as (N x M) column-major, ld = N
        if (cublasLtMatrixLayoutCreate(&mADesc, ioType, mN, mK, mN) != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }
        if (cublasLtMatrixLayoutCreate(&mBDesc, ioType, mK, mM, mK) != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }
        if (cublasLtMatrixLayoutCreate(&mCDesc, ioType, mN, mM, mN) != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }

        cublasLtMatmulPreference_t preference{};
        if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
        {
            return -1;
        }

        size_t workspaceSize = kWORKSPACE_SIZE;
        if (cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize))
            != CUBLAS_STATUS_SUCCESS)
        {
            cublasLtMatmulPreferenceDestroy(preference);
            return -1;
        }

        cublasLtMatmulHeuristicResult_t heuristic{};
        int32_t returnedResults = 0;
        cublasStatus_t const status = cublasLtMatmulAlgoGetHeuristic(
            mLtHandle, mOperationDesc, mADesc, mBDesc, mCDesc, mCDesc, preference, 1, &heuristic, &returnedResults);
        cublasLtMatmulPreferenceDestroy(preference);
        if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0)
        {
            return -1;
        }

        mAlgo = heuristic.algo;
        mAlgoAvailable = true;
        return 0;
    }

    DataType mType{DataType::kHALF};
    cublasComputeType_t mComputeType{CUBLAS_COMPUTE_16F};
    int64_t mM{0};
    int64_t mN{0};
    int64_t mK{0};
    bool mAlgoAvailable{false};

    cublasLtHandle_t mLtHandle{nullptr};
    cublasLtMatmulDesc_t mOperationDesc{nullptr};
    cublasLtMatrixLayout_t mADesc{nullptr};
    cublasLtMatrixLayout_t mBDesc{nullptr};
    cublasLtMatrixLayout_t mCDesc{nullptr};
    cublasLtMatmulAlgo_t mAlgo{};

    PluginFieldCollection mFC{};
};

class ProjectionMatmulPluginCreator : public IPluginCreatorV3One
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
        return new ProjectionMatmulPlugin();
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

private:
    PluginFieldCollection mFC{};
};

} // namespace

REGISTER_TENSORRT_PLUGIN(ProjectionMatmulPluginCreator);
