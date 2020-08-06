#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Rmsprop::optimize(Layer::Abstract *layer) {
                    Computation::factory().gradientRmsProp(layer, this->learningRate, this->batchSize);
                }
            }
        }
    }
}