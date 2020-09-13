#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Adagrad::optimize(Layer::Abstract *layer) {
                    layer->getComputation()->gradientAdagrad(this->learningRate, this->batchSize);
                }
            }
        }
    }
}