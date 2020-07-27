#include "../include.h"
#include "../common.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            ConvBuilder::ConvBuilder(T_Dimension dims) : Abstract<Network::ConvNetwork>(dims) {

            }

            void ConvBuilder::firstLayerTransition(Layer::LayerPointer layer) {
                layer->setSize(this->dimension.width,
                               this->dimension.height,
                               this->dimension.depth);
            }

            ConvBuilder ConvBuilder::fromJSON(T_String path) {
                std::ifstream fileStream(path);
                nlohmann::json jsonFile;

                fileStream >> jsonFile;
                fileStream.close();

                std::vector<T_Size> inputSize = jsonFile["inputSize"];

                T_Dimension dimension;
                dimension.width = inputSize.at(0);
                dimension.height = inputSize.at(1);
                dimension.depth = inputSize.at(2);

                ConvBuilder builder(dimension);

                nlohmann::json savedLayers = jsonFile["layers"];

                for (auto &element : savedLayers) {
                    T_String layerType = element["type"];

                    if (layerType == Layer::TYPE_LOGISTIC ||
                        layerType == Layer::TYPE_RELU ||
                        layerType == Layer::TYPE_PURELIN ||
                        layerType == Layer::TYPE_SOFTMAX) {

                        T_Size size = element["size"];

                        if (layerType == Layer::TYPE_LOGISTIC) {
                            builder.createLayer<Layer::Logistic>([&size](auto *layer) {
                                layer->setSize(size);
                            });
                        } else if (layerType == Layer::TYPE_RELU) {
                            builder.createLayer<Layer::Relu>([&size](auto *layer) {
                                layer->setSize(size);
                            });
                        } else if (layerType == Layer::TYPE_SOFTMAX) {
                            builder.createLayer<Layer::Softmax>([&size](auto *layer) {
                                layer->setSize(size);
                            });
                        } else if (layerType == Layer::TYPE_PURELIN) {
                            builder.createLayer<Layer::Purelin>([&size](auto *layer) {
                                layer->setSize(size);
                            });
                        }
                    } else if (layerType == Layer::TYPE_CONV) {
                        T_Size filterSize = element["filterSize"];
                        T_Size stride = element["stride"];
                        T_Size numFilters = element["numFilters"];
                        T_Size padding = element["padding"];

                        builder.createLayer<Layer::Conv>([&filterSize, &stride, &numFilters, &padding](auto *layer) {
                            layer->setFilterSize(filterSize);
                            layer->setStride(stride);
                            layer->setNumFilters(numFilters);
                            layer->setPadding(padding);
                        });
                    } else if (layerType == Layer::TYPE_MAXPOOL) {
                        T_Size stride = element["stride"];
                        T_Size filterSize = element["filterSize"];

                        builder.createLayer<Layer::MaxPool>([&stride, &filterSize](auto *layer) {
                            layer->setStride(stride);
                            layer->setFilterSize(filterSize);
                        });
                    } else if (layerType == Layer::TYPE_FULLYCONNECTED) {
                        builder.createLayer<Layer::FullyConnected>([](auto *layer) {
                            // empty
                        });
                    }
                }

                Math::T_RawVector theta = jsonFile["weights"];
                builder.getNetwork().setRolledTheta(Math::rawToVector(theta));

                return builder;
            }
        }
    }
}
