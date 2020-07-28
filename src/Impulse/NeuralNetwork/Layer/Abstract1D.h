#pragma once

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract1D : public Abstract {
            public:
                Abstract1D();

                void configure() override;

                bool is1D() override;

                bool is3D() override;

                void transition(Layer::LayerPointer prevLayer) override;
            };
        }
    }
}
