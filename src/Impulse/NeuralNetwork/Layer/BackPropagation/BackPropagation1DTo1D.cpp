#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagation1DTo1D::BackPropagation1DTo1D
                        (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) :
                        Abstract(layer, previousLayer) {

                }

                Eigen::MatrixXd BackPropagation1DTo1D::propagate(const Eigen::MatrixXd &input,
                                                                 T_Size numberOfExamples,
                                                                 double regularization,
                                                                 const Eigen::MatrixXd &sigma) {

                    Eigen::MatrixXd previousActivations =
                            this->previousLayer == nullptr ? input : this->previousLayer->A;

                    Eigen::MatrixXd delta = sigma * previousActivations.transpose().conjugate();

                    this->layer->gW = delta.array() / numberOfExamples +
                                      (regularization / numberOfExamples * this->layer->W.array());
                    this->layer->gB = sigma.rowwise().sum() / numberOfExamples;

                    if (this->previousLayer != nullptr) {
                        Eigen::MatrixXd tmp1 = this->layer->W.transpose() * sigma;
                        Eigen::MatrixXd tmp2 = this->previousLayer->derivative(this->previousLayer->A);

                        return tmp1.array() * tmp2.array();
                    }
                    return Eigen::MatrixXd(); // return empty - this is first layer
                }
            }
        }
    }
}