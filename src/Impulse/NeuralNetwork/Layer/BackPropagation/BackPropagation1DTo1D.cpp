#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagation1DTo1D::BackPropagation1DTo1D
                        (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) :
                        Abstract(layer, previousLayer) {

                }

                Math::T_Matrix BackPropagation1DTo1D::propagate(const Math::T_Matrix &input,
                                                                T_Size numberOfExamples,
                                                                double regularization,
                                                                const Math::T_Matrix &sigma) {

                    Math::T_Matrix previousActivations =
                            this->previousLayer == nullptr ? input : this->previousLayer->A;

                    Math::T_Matrix delta = sigma * previousActivations.transpose().conjugate();

                    this->layer->gW = delta.array() / numberOfExamples +
                                      (regularization / numberOfExamples * this->layer->W.array());
                    this->layer->gb = sigma.rowwise().sum() / numberOfExamples;

                    if (this->previousLayer != nullptr) {
                        Math::T_Matrix tmp1 = this->layer->W.transpose() * sigma;
                        Math::T_Matrix tmp2 = this->previousLayer->derivative();

                        return tmp1.array() * tmp2.array();
                    }
                    return Math::T_Matrix(); // return empty - this is first layer
                }
            }
        }
    }
}