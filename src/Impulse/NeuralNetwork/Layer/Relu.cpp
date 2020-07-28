#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Relu::Relu() : Abstract1D() {};

            Eigen::MatrixXd Relu::activation(Eigen::MatrixXd &m) {
                Eigen::MatrixXd result = m.unaryExpr([](const double x) {
                    return std::max(0.0, x);
                });
                return result;
            }

            Eigen::MatrixXd Relu::derivative() {
                return this->A.unaryExpr([](const double x) {
                    if (x < 0.0) {
                        return 0.0;
                    }
                    return 1.0;
                });
            }

            const T_String Relu::getType() {
                return TYPE_RELU;
            }

            double Relu::loss(Eigen::MatrixXd output, Eigen::MatrixXd predictions) {
                // TODO
                return 0.0;
            }

            double Relu::error(T_Size m) {
                return 0.0; // TODO
            }
        }
    }
}
