#ifndef IMPULSE_NEURALNETWORK_LAYER_POOL_H
#define IMPULSE_NEURALNETWORK_LAYER_POOL_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_MAXPOOL = "maxpool";

            class MaxPool : public Abstract3D {
            protected:
                T_Size filterSize = 2;
                T_Size stride = 2;
            public:
                MaxPool();

                void configure() override;

                void setFilterSize(T_Size value);

                T_Size getFilterSize();

                void setStride(T_Size value);

                T_Size getStride();

                Math::T_Matrix forward(const Math::T_Matrix &input) override;

                Math::T_Matrix activation(Math::T_Matrix &m) override;

                Math::T_Matrix derivative() override;

                const T_String getType() override;

                double loss(Math::T_Matrix output, Math::T_Matrix predictions) override;

                double error(T_Size m) override;

                T_Size getOutputHeight() override;

                T_Size getOutputWidth() override;

                T_Size getOutputDepth() override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_POOL_H
