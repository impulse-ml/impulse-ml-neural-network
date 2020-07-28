#pragma once

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
