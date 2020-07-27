#ifndef IMPULSE_NEURALNETWORK_CONV_BUILDER_H
#define IMPULSE_NEURALNETWORK_CONV_BUILDER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class ConvBuilder : public Abstract<Network::ConvNetwork> {
            protected:
            public:
                explicit ConvBuilder(T_Dimension dims);

                void firstLayerTransition(Layer::LayerPointer layer) override;

                static ConvBuilder fromJSON(T_String path);
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_CONV_BUILDER_H
