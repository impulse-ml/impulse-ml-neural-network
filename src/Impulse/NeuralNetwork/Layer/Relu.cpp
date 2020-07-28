#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Relu::Relu() : Abstract1D() {};

            Eigen::MatrixXd Relu::activation(Eigen::MatrixXd &m) {
                return Computation::factory().reluActivation(m);
            }

            Eigen::MatrixXd Relu::derivative() {
                return Computation::factory().reluDerivative(this->A);
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
