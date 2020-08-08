#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            template<class OPTIMIZER_TYPE>
            Stochastic<OPTIMIZER_TYPE>::Stochastic(Network::Abstract &net) : AbstractTrainer<OPTIMIZER_TYPE>(net) {}

            template
            Stochastic<Optimizer::GradientDescent>::Stochastic(Network::Abstract &net);

            template
            Stochastic<Optimizer::Adam>::Stochastic(Network::Abstract &net);

            template
            Stochastic<Optimizer::Adagrad>::Stochastic(Network::Abstract &net);

            template
            Stochastic<Optimizer::Momentum>::Stochastic(Network::Abstract &net);

            template
            Stochastic<Optimizer::Nesterov>::Stochastic(Network::Abstract &net);

            template
            Stochastic<Optimizer::Rmsprop>::Stochastic(Network::Abstract &net);

            template
            Stochastic<Optimizer::Adadelta>::Stochastic(Network::Abstract &net);

            template<class OPTIMIZER_TYPE>
            void Stochastic<OPTIMIZER_TYPE>::train(Impulse::Dataset::SlicedDataset &dataSet) {
                T_Size t = 0;

                for (T_Size i = 0; i < this->learningIterations; i++) {
                    high_resolution_clock::time_point begin = high_resolution_clock::now();

                    Eigen::MatrixXd input = dataSet.getInput();
                    Eigen::MatrixXd output = dataSet.getOutput();
                    Eigen::MatrixXd forward = this->network.forward(input);

                    this->network.backward(input, output, forward, this->regularization);

                    Trainer::CostGradientResult result = this->cost(dataSet);

                    for (T_Size j = 0; j < this->network.getSize(); j++) {
                        Layer::LayerPointer layer = this->network.getLayer(j);
                        if (layer->getType() == Layer::TYPE_MAXPOOL) {
                            continue;
                        }
                        this->optimizer->setT(++t);
                        this->optimizer->optimize(layer.get());
                    }

                    Trainer::CostGradientResult currentResult = this->cost(dataSet);

                    if (this->verbose) {
                        if ((i + 1) % this->verboseStep == 0) {
                            high_resolution_clock::time_point end = high_resolution_clock::now();
                            auto duration = duration_cast<milliseconds>(end - begin).count();
                            std::cout << "Iteration: " << (i + 1)
                                      << " | Cost: " << currentResult.getCost()
                                      << " | Accuracy: " << currentResult.getAccuracy()
                                      << "% | Time: " << duration
                                      << std::endl;
                        }
                    }

                    if (this->stepCallbackSet) {
                        this->stepCallback();
                    }

                    if (currentResult.getCost() > result.getCost()) {
                        std::cout << "Terminated." << std::endl;
                        break;
                    }
                }
            }
        }
    }
}
