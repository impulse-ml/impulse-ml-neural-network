#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Logistic::Logistic() : Abstract1D() {};

            Eigen::MatrixXd Logistic::activation(Eigen::MatrixXd &m) {
                return Computation::factory().logisticActivation(m);
            }

            Eigen::MatrixXd Logistic::derivative(Eigen::MatrixXd &a) {
                return Computation::factory().logisticDerivative(a);
            }

            const T_String Logistic::getType() {
                return TYPE_LOGISTIC;
            }

            double Logistic::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                return Computation::factory().logisticLoss(output, predictions);
            }

            double Logistic::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
