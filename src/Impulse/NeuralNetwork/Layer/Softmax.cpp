#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Softmax::Softmax() : Abstract1D() {}

            Eigen::MatrixXd Softmax::activation(Eigen::MatrixXd &m) {
                return ComputationCpu::factory().softmaxActivation(m);
            }

            Eigen::MatrixXd Softmax::derivative() {
                return ComputationCpu::factory().softmaxDerivative(this->A);
            }

            const T_String Softmax::getType() {
                return TYPE_SOFTMAX;
            }

            double Softmax::loss(Eigen::MatrixXd & output, Eigen::MatrixXd & predictions) {
                return ComputationCpu::factory().softmaxLoss(output, predictions);
            }

            double Softmax::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
