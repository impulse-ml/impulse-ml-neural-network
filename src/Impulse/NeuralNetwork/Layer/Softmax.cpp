#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Softmax::Softmax() : Abstract1D() {}

            Eigen::MatrixXd Softmax::activation(Eigen::MatrixXd &m) {
                Eigen::MatrixXd t = m.unaryExpr([](const double x) {
                    return exp(x);
                });
                Eigen::MatrixXd divider = t.colwise().sum().replicate(t.rows(), 1);
                Eigen::MatrixXd result = t.array() / divider.array();
                return result;
            }

            Eigen::MatrixXd Softmax::derivative() {
                // TODO
                return Eigen::MatrixXd();
            }

            const T_String Softmax::getType() {
                return TYPE_SOFTMAX;
            }

            double Softmax::loss(Eigen::MatrixXd output, Eigen::MatrixXd predictions) {
                Eigen::MatrixXd loss = (output.array() *
                                       predictions.unaryExpr([](const double x) { return log(x); }).array());
                return loss.sum();
            }

            double Softmax::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
