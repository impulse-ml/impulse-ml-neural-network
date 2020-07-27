#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagation3DTo1D::BackPropagation3DTo1D
                        (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) :
                        Abstract(layer, previousLayer) {

                }

                Math::T_Matrix BackPropagation3DTo1D::propagate(const Math::T_Matrix &input,
                                                                T_Size numberOfExamples,
                                                                double regularization,
                                                                const Math::T_Matrix &sigma) {

                    return sigma;
                }
            }
        }
    }
}