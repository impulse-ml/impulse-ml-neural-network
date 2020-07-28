#pragma once

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
