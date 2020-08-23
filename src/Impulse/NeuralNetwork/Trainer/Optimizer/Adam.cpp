#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Adam::optimize(Layer::Abstract *layer) {
                    layer->getComputation()->gradientAdam(this->learningRate, this->t);
                }
            }
        }
    }
}