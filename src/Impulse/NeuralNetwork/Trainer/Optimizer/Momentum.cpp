#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Momentum::optimize(Layer::Abstract *layer) {
                    Computation::factory().gradientMomentum(layer, this->learningRate, this->batchSize);
                }
            }
        }
    }
}