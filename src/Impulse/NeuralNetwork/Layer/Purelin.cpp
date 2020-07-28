#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Purelin::Purelin() : Abstract1D() {};

            Eigen::MatrixXd Purelin::activation(Eigen::MatrixXd &m) {
                return m;
            }

            Eigen::MatrixXd Purelin::derivative() {
                Eigen::MatrixXd d(this->Z.rows(), this->Z.cols());
                d.setOnes();
                return d;
            }

            const T_String Purelin::getType() {
                return TYPE_PURELIN;
            }

            double Purelin::loss(Eigen::MatrixXd & output, Eigen::MatrixXd & predictions) {
                Eigen::MatrixXd loss = (predictions.array() - output.array()).unaryExpr([](const double x) {
                    return pow(x, 2.0);
                });
                return loss.sum();
            }

            double Purelin::error(T_Size m) {
                return (1.0 / (2.0 * (double) m));
            }
        }
    }
}
