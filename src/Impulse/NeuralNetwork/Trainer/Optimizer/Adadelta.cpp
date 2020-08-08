#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Adadelta::optimize(Layer::Abstract *layer) {
                    Computation::factory().gradientAdadelta(layer, this->learningRate, this->batchSize);
                }
            }
        }
    }
}