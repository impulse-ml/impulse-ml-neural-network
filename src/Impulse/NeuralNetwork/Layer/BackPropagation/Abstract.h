#pragma once

#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            // fwd declarations
            class Abstract;

            typedef std::shared_ptr<Abstract> LayerPointer;

            namespace BackPropagation {

                // fwd declaration
                class Abstract;

                typedef std::shared_ptr<Abstract> BackPropagationPointer;

                class Abstract {
                protected:
                    Layer::LayerPointer layer;
                    Layer::LayerPointer previousLayer;
                public:
                    Abstract(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    virtual Eigen::MatrixXd propagate(const Eigen::MatrixXd &input,
                                                      T_Size numberOfExamples,
                                                      double regularization,
                                                      const Eigen::MatrixXd &delta) = 0;
                };
            }
        }
    }
}
