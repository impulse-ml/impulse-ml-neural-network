#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Relu::Relu() : Abstract1D() {};

            Eigen::MatrixXd Relu::activation() {
                this->computation->reluActivation();
                return this->computation->getVariable("A");
            }

            Eigen::MatrixXd Relu::derivative(Eigen::MatrixXd &a) {
                return this->computation->reluDerivative(a);
            }

            const T_String Relu::getType() {
                return TYPE_RELU;
            }

            double Relu::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                // TODO
                return 0.0;
            }

            double Relu::error(T_Size m) {
                // TODO
                return 0.0;
            }
        }
    }
}
