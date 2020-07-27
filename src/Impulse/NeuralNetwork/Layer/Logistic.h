#ifndef IMPULSE_NEURALNETWORK_LAYER_LOGISTIC_H
#define IMPULSE_NEURALNETWORK_LAYER_LOGISTIC_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_LOGISTIC = "logistic";

            class Logistic : public Abstract1D {
            protected:
            public:
                Logistic();

                Math::T_Matrix activation(Math::T_Matrix &m) override;

                Math::T_Matrix derivative() override;

                const T_String getType() override;

                double loss(Math::T_Matrix output, Math::T_Matrix predictions) override;

                double error(T_Size m) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_LOGISTIC_H
