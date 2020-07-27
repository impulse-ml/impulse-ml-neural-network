#ifndef IMPULSE_NEURALNETWORK_NEW_BUILDER_H
#define IMPULSE_NEURALNETWORK_NEW_BUILDER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            template<class NETWORK_TYPE>
            class Abstract {
            protected:
                NETWORK_TYPE network;
                T_Dimension dimension;
                Layer::LayerPointer previousLayer = nullptr;
            public:
                explicit Abstract(T_Dimension dims);

                template<typename LAYER_TYPE>
                void createLayer(std::function<void(LAYER_TYPE *)> callback);

                NETWORK_TYPE &getNetwork();

                virtual void firstLayerTransition(Layer::LayerPointer layer) = 0;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_NEW_BUILDER_H
