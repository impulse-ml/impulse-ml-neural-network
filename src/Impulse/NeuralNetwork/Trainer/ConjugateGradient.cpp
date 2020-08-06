#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            template<class OPTIMIZER_TYPE>
            ConjugateGradient<OPTIMIZER_TYPE>::ConjugateGradient(Network::Abstract &net)
                    : AbstractTrainer<OPTIMIZER_TYPE>(net) {}

            template<class OPTIMIZER_TYPE>
            void ConjugateGradient<OPTIMIZER_TYPE>::train(Impulse::Dataset::SlicedDataset &dataSet) {
                Math::Fmincg minimizer;
                Eigen::VectorXd theta = this->network.getRolledTheta();
                double regularization = this->regularization;

                Eigen::MatrixXd input = dataSet.getInput();
                Eigen::MatrixXd output = dataSet.getOutput();
                Eigen::MatrixXd forward = this->network.forward(input);

                this->network.backward(input, output, forward, this->regularization);

                Trainer::StepFunction callback(
                        [this, &dataSet, &regularization, &input, &output, &forward](Eigen::VectorXd input2) {
                            this->network.setRolledTheta(input2);
                            Eigen::MatrixXd forward = this->network.forward(input);
                            this->network.backward(input, output, forward, regularization);
                            return this->cost(dataSet, true);
                        });

                Eigen::VectorXd minimized = minimizer.minimize(callback, theta, this->learningIterations,
                                                               this->verbose);
                this->network.setRolledTheta(minimized);
            }
        }
    }
}
