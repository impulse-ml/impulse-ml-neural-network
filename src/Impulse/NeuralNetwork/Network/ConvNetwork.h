#ifndef IMPULSE_NEURALNETWORK_CONV_NETWORK_H
#define IMPULSE_NEURALNETWORK_CONV_NETWORK_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            class ConvNetwork : public Abstract {
            public:
                explicit ConvNetwork(T_Dimension dim);
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_CONV_NETWORK_H
