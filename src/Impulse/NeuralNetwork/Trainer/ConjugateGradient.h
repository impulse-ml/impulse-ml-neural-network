#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            template<class OPTIMIZER_TYPE>
            class ConjugateGradient : public AbstractTrainer<OPTIMIZER_TYPE> {
            public:
                explicit ConjugateGradient(Network::Abstract &net);

                void train(Impulse::Dataset::SlicedDataset &dataSet) override;
            };
        }
    }
}
