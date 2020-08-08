#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Tanh::Tanh() : Abstract1D() {};

            Eigen::MatrixXd Tanh::activation(Eigen::MatrixXd &m) {
                return Computation::factory().tanhActivation(m);
            }

            Eigen::MatrixXd Tanh::derivative(Eigen::MatrixXd &a) {
                return Computation::factory().tanhDerivative(a);
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
