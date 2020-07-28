#pragma once

#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                class BackPropagationToMaxPool : public Abstract {
                public:
                    BackPropagationToMaxPool(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    Eigen::MatrixXd propagate(const Eigen::MatrixXd &input,
                                              T_Size numberOfExamples,
                                              double regularization,
                                              const Eigen::MatrixXd &sigma) override;
                };
            }
        }
    }
}
