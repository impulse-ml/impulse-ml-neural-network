#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                class Momentum : public Abstract {
                public:
                    void optimize(Layer::Abstract *);
                };
            }
        }
    }
}