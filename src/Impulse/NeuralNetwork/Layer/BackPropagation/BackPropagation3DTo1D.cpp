#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagation3DTo1D::BackPropagation3DTo1D
                        (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) :
                        Abstract(layer, previousLayer) {

                }

                Eigen::MatrixXd BackPropagation3DTo1D::propagate(const Eigen::MatrixXd &input,
                                                                 T_Size numberOfExamples,
                                                                 double regularization,
                                                                 const Eigen::MatrixXd &sigma) {

                    return sigma;
                }
            }
        }
    }
}