#include "NvInferRuntime.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace
{

using namespace nvinfer1;

char constexpr kPLUGIN_NAME[] = "RfProbeIdentityCopy";
char constexpr kPLUGIN_VERSION[] = "1";

size_t getElementSize(DataType type) noexcept
{
    switch (type)
    {
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT32: return 4;
    case DataType::kINT64: return 8;
    case DataType::kINT8: return 1;
    case DataType::kBOOL: return 1;
    default: return 0;
    }
}

size_t getNumElements(Dims const& dims) noexcept
{
    size_t volume = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        volume *= static_cast<size_t>(dims.d[i]);
    }
    return volume;
}

class IdentityCopyPlugin : public IPluginV3,
                           public IPluginV3OneCore,
                           public IPluginV3OneBuild,
                           public IPluginV3OneRuntime
{
public:
    IdentityCopyPlugin() = default;
    IdentityCopyPlugin(IdentityCopyPlugin const&) = default;

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
        return new IdentityCopyPlugin(*this);
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

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 1 || nbOutputs != 1)
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

        return linear && desc.type == inOut[0].desc.type && desc.format == inOut[0].desc.format;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        outputs[0].nbDims = inputs[0].nbDims;
        for (int32_t i = 0; i < inputs[0].nbDims; ++i)
        {
            outputs[0].d[i] = inputs[0].d[i];
        }
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const
        noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        size_t const elementSize = getElementSize(inputDesc[0].type);
        if (elementSize == 0)
        {
            return -1;
        }

        size_t const numBytes = getNumElements(inputDesc[0].dims) * elementSize;
        cudaError_t const status =
            cudaMemcpyAsync(outputs[0], inputs[0], numBytes, cudaMemcpyDeviceToDevice, stream);
        return status == cudaSuccess ? 0 : -1;
    }

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mFC;
    }

private:
    PluginFieldCollection mFC{};
};

class IdentityCopyPluginCreator : public IPluginCreatorV3One
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
        return new IdentityCopyPlugin();
    }

    char const* getPluginNamespace() const noexcept override
    {
        return "";
    }

private:
    PluginFieldCollection mFC{};
};

} // namespace

REGISTER_TENSORRT_PLUGIN(IdentityCopyPluginCreator);
