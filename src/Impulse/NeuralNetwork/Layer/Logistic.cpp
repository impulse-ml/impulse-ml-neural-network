#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Logistic::Logistic() : Abstract1D() {};

            Math::T_Matrix Logistic::activation(Math::T_Matrix &m) {
                return m.unaryExpr([](const double x) {
                    return 1.0 / (1.0 + exp(-x));
                });
            }

            Math::T_Matrix Logistic::derivative() {
                return this->A.array() * (1.0 - this->A.array());
            }

            const T_String Logistic::getType() {
                return TYPE_LOGISTIC;
            }

            double Logistic::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix loss =
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
