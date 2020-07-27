#ifndef IMPULSE_NEURALNETWORK_TRAINER_GRADIENT_DESCENT_TRAINER_H
#define IMPULSE_NEURALNETWORK_TRAINER_GRADIENT_DESCENT_TRAINER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class GradientDescent : public AbstractTrainer {
            public:
                explicit GradientDescent(Network::Abstract &net);

                void train(Impulse::Dataset::SlicedDataset &dataSet) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_TRAINER_GRADIENT_DESCENT_TRAINER_H
