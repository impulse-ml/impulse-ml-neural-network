#pragma once

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                class BackPropagation3DTo1D : public Abstract {
                public:
                    BackPropagation3DTo1D(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    Eigen::MatrixXd propagate(const Eigen::MatrixXd &input,
                                              T_Size numberOfExamples,
                                              double regularization,
                                              const Eigen::MatrixXd &sigma);
                };
            }
        }
    }
}
