#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Nesterov::optimize(Layer::Abstract *layer) {
                    Computation::factory().gradientNesterov(layer, this->learningRate, this->batchSize);
                }
            }
        }
    }
}