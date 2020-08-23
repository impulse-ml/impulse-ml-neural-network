#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Rmsprop::optimize(Layer::Abstract *layer) {
                    layer->getComputation()->gradientRmsProp(this->learningRate, this->batchSize);
                }
            }
        }
    }
}