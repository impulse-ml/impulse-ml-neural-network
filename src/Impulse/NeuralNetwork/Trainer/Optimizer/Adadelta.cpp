#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Adadelta::optimize(Layer::Abstract *layer) {
                    layer->getComputation()->gradientAdadelta(this->learningRate, this->batchSize);
                }
            }
        }
    }
}