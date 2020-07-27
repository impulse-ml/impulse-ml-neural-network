#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Serializer::Serializer(Network::Abstract &net) : network(net) {}

        void Serializer::toJSON(T_String path) {
            nlohmann::json result;

            T_Dimension dim = this->network.getDimension();
            result["inputSize"] = {dim.width, dim.height, dim.depth};

            result["layers"] = {};
            for (T_Size i = 0; i < this->network.getSize(); i++) {
                Layer::LayerPointer layer = this->network.getLayer(i);
                result["layers"][i] = nlohmann::json::object();
                result["layers"][i]["type"] = layer->getType();

                if (layer->getType() == Layer::TYPE_MAXPOOL) {
                    auto *_layer = (Layer::MaxPool *) layer.get();
                    result["layers"][i]["filterSize"] = _layer->getFilterSize();
                    result["layers"][i]["stride"] = _layer->getStride();
                } else if (layer->getType() == Layer::TYPE_CONV) {
                    auto *_layer = (Layer::Conv *) layer.get();
                    result["layers"][i]["filterSize"] = _layer->getFilterSize();
                    result["layers"][i]["stride"] = _layer->getStride();
                    result["layers"][i]["numFilters"] = _layer->getNumFilters();
                    result["layers"][i]["padding"] = _layer->getPadding();
                } else if (layer->getType() == Layer::TYPE_LOGISTIC ||
                           layer->getType() == Layer::TYPE_PURELIN ||
                           layer->getType() == Layer::TYPE_RELU ||
                           layer->getType() == Layer::TYPE_SOFTMAX) {
                    result["layers"][i]["size"] = layer->getSize();
                } else if (layer->getType() == Layer::TYPE_FULLYCONNECTED) {
                    // no dump
                }
            }

            Math::T_Vector weights = this->network.getRolledTheta();
            result["weights"] = Math::vectorToRaw(weights);

            std::ofstream out(path);
            out << result.dump();
            out.close();
        }
    }
}
