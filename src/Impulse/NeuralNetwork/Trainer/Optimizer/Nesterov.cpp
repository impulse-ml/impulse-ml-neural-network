#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Nesterov::optimize(Layer::Abstract *layer) {
                    layer->getComputation()->gradientNesterov(this->learningRate, this->batchSize);
                }
            }
        }
    }
}