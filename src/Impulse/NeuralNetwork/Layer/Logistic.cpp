#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Logistic::Logistic() : Abstract1D() {};

            Eigen::MatrixXd Logistic::activation(Eigen::MatrixXd &m) {
                return m.unaryExpr([](const double x) {
                    return 1.0 / (1.0 + exp(-x));
                });
            }

            Eigen::MatrixXd Logistic::derivative() {
                return this->A.array() * (1.0 - this->A.array());
            }

            const T_String Logistic::getType() {
                return TYPE_LOGISTIC;
            }

            double Logistic::loss(Eigen::MatrixXd & output, Eigen::MatrixXd & predictions) {
                Eigen::MatrixXd loss =
                        (output.array() * predictions.unaryExpr([](const double x) { return log(x); }).array())
                        +
                        (output.unaryExpr([](const double x) { return 1.0 - x; }).array()
                         *
                         predictions.unaryExpr([](const double x) { return log(1.0 - x); }).array()
                        );
                return loss.sum();
            }

            double Logistic::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
