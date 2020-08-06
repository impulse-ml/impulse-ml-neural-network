#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            template<class OPTIMIZER_TYPE>
            class MiniBatch : public AbstractTrainer<OPTIMIZER_TYPE> {
            protected:
                T_Size batchSize = 100;
            public:
                explicit MiniBatch<OPTIMIZER_TYPE>(Network::Abstract &net);

                void setBatchSize(T_Size value);

                void train(Impulse::Dataset::SlicedDataset &dataSet);
            };
        }
    }
}
