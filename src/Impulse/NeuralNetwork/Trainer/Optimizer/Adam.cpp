#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Adam::optimize(Layer::Abstract *layer) {
                    Computation::factory().gradientAdam(layer, this->learningRate, this->t);
                }
            }
        }
    }
}