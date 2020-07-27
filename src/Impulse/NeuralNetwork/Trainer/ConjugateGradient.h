#ifndef IMPULSE_NEURALNETWORK_TRAINER_CONJUGATE_GRADIENT_TRAINER_H
#define IMPULSE_NEURALNETWORK_TRAINER_CONJUGATE_GRADIENT_TRAINER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class ConjugateGradient : public AbstractTrainer {
            public:
                explicit ConjugateGradient(Network::Abstract &net);

                void train(Impulse::Dataset::SlicedDataset &dataSet) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_TRAINER_CONJUGATE_GRADIENT_TRAINER_H
