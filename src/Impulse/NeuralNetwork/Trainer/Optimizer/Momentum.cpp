#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Momentum::optimize(Layer::Abstract *layer) {
                    layer->getComputation()->gradientMomentum(this->learningRate, this->batchSize);
                }
            }
        }
    }
}