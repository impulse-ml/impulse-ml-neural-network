#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Softmax::Softmax() : Abstract1D() {}

            Eigen::MatrixXd Softmax::activation() {
                this->computation->softmaxActivation();
                return this->computation->getVariable("A");
            }

            Eigen::MatrixXd Softmax::derivative(Eigen::MatrixXd &a) {
                return this->computation->softmaxDerivative(a);
            }

            const T_String Softmax::getType() {
                return TYPE_SOFTMAX;
            }

            double Softmax::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                return this->computation->softmaxLoss(output, predictions);
            }

            double Softmax::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
