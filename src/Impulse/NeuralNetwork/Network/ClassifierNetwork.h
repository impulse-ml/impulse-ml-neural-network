#ifndef IMPULSE_NEURALNETWORK_CLASSIFIER_NETWORK_H
#define IMPULSE_NEURALNETWORK_CLASSIFIER_NETWORK_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            class ClassifierNetwork : public Abstract {
            public:
                explicit ClassifierNetwork(T_Dimension dim);
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_CLASSIFIER_NETWORK_H
