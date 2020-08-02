#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            ClassifierBuilder::ClassifierBuilder(T_Dimension dims) : Abstract<Network::ClassifierNetwork>(dims) {
            }

            void ClassifierBuilder::firstLayerTransition(Layer::LayerPointer layer) {
                layer->setPrevSize(this->dimension.width);
            }

            ClassifierBuilder ClassifierBuilder::fromJSON(T_String path) {
                std::ifstream fileStream(path);
                nlohmann::json jsonFile;

                fileStream >> jsonFile;
                fileStream.close();

                std::vector<T_Size> inputSize = jsonFile["inputSize"];

                ClassifierBuilder builder({inputSize[0]});

                nlohmann::json savedLayers = jsonFile["layers"];

                for (auto &element : savedLayers) {
                    T_String layerType = element["type"];

                    if (layerType == Layer::TYPE_LOGISTIC ||
                        layerType == Layer::TYPE_RELU ||
                        layerType == Layer::TYPE_PURELIN ||
                        layerType == Layer::TYPE_SOFTMAX ||
                        layerType == Layer::TYPE_TANH ||
                        layerType == Layer::TYPE_SOFTPLUS) {

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
                        } else if (layerType == Layer::TYPE_TANH) {
                            builder.createLayer<Layer::Tanh>([&size](auto *layer) {
                                layer->setSize(size);
                            });
                        } else if (layerType == Layer::TYPE_SOFTPLUS) {
                            builder.createLayer<Layer::Softplus>([&size](auto *layer) {
                                layer->setSize(size);
                            });
                        }
                    }
                }

                Math::T_RawVector weights = jsonFile["weights"];
                Eigen::VectorXd theta = Math::rawToVector(weights);
                builder.getNetwork().setRolledTheta(theta);

                return builder;
            }
        }
    }
}
