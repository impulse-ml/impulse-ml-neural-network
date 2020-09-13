#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Purelin::Purelin() : Abstract1D() {};

            Eigen::MatrixXd Purelin::activation() {
                return this->computation->getVariable("Z");
            }

            Eigen::MatrixXd Purelin::derivative(Eigen::MatrixXd &) {
                Eigen::MatrixXd d(this->computation->getVariable("Z").rows(),
                                  this->computation->getVariable("Z").cols());
                d.setOnes();
                return d;
            }

            const T_String Purelin::getType() {
                return TYPE_PURELIN;
            }

            double Purelin::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                return this->computation->purelinLoss(output, predictions);
            }

            double Purelin::error(T_Size m) {
                return (1.0 / (2.0 * (double) m));
            }
        }
    }
}
