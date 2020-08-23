#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Tanh::Tanh() : Abstract1D() {};

            Eigen::MatrixXd Tanh::activation() {
                this->computation->tanhActivation();
                return this->computation->getVariable("A");
            }

            Eigen::MatrixXd Tanh::derivative(Eigen::MatrixXd &a) {
                return this->computation->tanhDerivative(a);
            }

            const T_String Tanh::getType() {
                return TYPE_TANH;
            }

            double Tanh::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                return 0.0; // TODO
            }

            double Tanh::error(T_Size m) {
                return 0.0; // TODO
            }
        }
    }
}
