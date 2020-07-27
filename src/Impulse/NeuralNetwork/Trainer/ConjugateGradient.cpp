#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            ConjugateGradient::ConjugateGradient(Network::Abstract &net) : AbstractTrainer(net) {}

            void ConjugateGradient::train(Impulse::Dataset::SlicedDataset &dataSet) {
                Math::Fmincg minimizer;
                Network::Abstract network = this->network;
                Math::T_Vector theta = network.getRolledTheta();
                double regularization = this->regularization;

                network.backward(dataSet.getInput(), dataSet.getOutput(), network.forward(dataSet.getInput()),
                                 this->regularization);

                Trainer::StepFunction callback(
                        [this, &dataSet, &regularization](Math::T_Vector input) {
                            this->network.setRolledTheta(input);
                            this->network.backward(dataSet.getInput(), dataSet.getOutput(),
                                                   this->network.forward(dataSet.getInput()), regularization);
                            return this->cost(dataSet, true);
                        });

                this->network.setRolledTheta(
                        minimizer.minimize(callback, theta, this->learningIterations, this->verbose));
            }
        }
    }
}
