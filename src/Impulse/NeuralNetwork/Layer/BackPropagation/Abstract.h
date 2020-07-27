#ifndef IMPULSE_VECTORIZED_ABSTRACT_H
#define IMPULSE_VECTORIZED_ABSTRACT_H

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

                    virtual Math::T_Matrix propagate(const Math::T_Matrix &input,
                                                     T_Size numberOfExamples,
                                                     double regularization,
                                                     const Math::T_Matrix &delta) = 0;
                };
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
