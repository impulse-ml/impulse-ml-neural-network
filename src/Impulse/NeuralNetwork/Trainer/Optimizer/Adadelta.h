#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                class Adadelta : public Abstract {
                public:
                    void optimize(Layer::Abstract *);
                };
            }
        }
    }
}