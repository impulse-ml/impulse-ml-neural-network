#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Adagrad::optimize(Layer::Abstract *layer) {
                    Computation::factory().gradientAdagrad(layer, this->learningRate, this->batchSize);
                }
            }
        }
    }
}