#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Softplus::Softplus() : Abstract1D() {};

            Eigen::MatrixXd Softplus::activation() {
                this->computation->softplusActivation();
                return this->computation->getVariable("A");
            }

            Eigen::MatrixXd Softplus::derivative(Eigen::MatrixXd &a) {
                return this->computation->softplusDerivative(a);
            }

            const T_String Softplus::getType() {
                return TYPE_SOFTPLUS;
            }

            double Softplus::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                return 0.0; // TODO
            }

            double Softplus::error(T_Size m) {
                return 0.0; // TODO
            }
        }
    }
}
