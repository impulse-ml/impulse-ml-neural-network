#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            template<typename OPTIMIZER_TYPE>
            AbstractTrainer<OPTIMIZER_TYPE>::AbstractTrainer(Network::Abstract &net) : network(net) {
                this->optimizer = new OPTIMIZER_TYPE;
            }

            template
            AbstractTrainer<Optimizer::Adam>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Adadelta>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Adagrad>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::GradientDescent>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Momentum>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Nesterov>::AbstractTrainer(Network::Abstract &net);

            template
            AbstractTrainer<Optimizer::Rmsprop>::AbstractTrainer(Network::Abstract &net);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setRegularization(double value) {
                this->regularization = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setRegularization(double value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setRegularization(double value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setLearningIterations(T_Size value) {
                this->learningIterations = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setLearningIterations(T_Size value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setLearningIterations(T_Size value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setLearningRate(double value) {
                this->learningRate = value;
                this->optimizer->setLearningRate(value);
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setLearningRate(double value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setLearningRate(double value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setVerbose(bool value) {
                this->verbose = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setVerbose(bool value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setVerbose(bool value);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setVerboseStep(int value) {
                this->verboseStep = value;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Momentum>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setVerboseStep(int value);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setVerboseStep(int value);

            template<typename OPTIMIZER_TYPE>
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<OPTIMIZER_TYPE>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient) {
                T_Size batchSize = 100;
                T_Size numberOfExamples = dataSet.getNumberOfExamples();
                auto numBatches = (T_Size) ceil((double) numberOfExamples / (double) batchSize);

                double cost = 0.0;
                double accuracy = 0.0;

                // calculate penalty
                double penalty = 0.0;

#pragma omp parallel
#pragma omp for
                for (T_Size i = 0; i < this->network.getSize(); i++) {
                    penalty += this->network.getLayer(i)->penalty();
                }

                // calculate cost from mini-batches
#pragma omp parallel
#pragma omp for
                for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                    Eigen::MatrixXd predictedOutput = this->network.forward(dataSet.getInput(offset, batchSize));
                    Eigen::MatrixXd correctOutput = dataSet.getOutput(offset, batchSize);

                    auto miniBatchSize = (T_Size) correctOutput.cols();

                    double loss = this->network.loss(correctOutput, predictedOutput); // loss for the mini-batch
                    double error = this->network.error(miniBatchSize); // error for the mini-batch

                    cost += (error * loss + ((this->regularization * penalty) / (2.0 * (double) miniBatchSize)))
                            /
                            // TODO: fix it
                            ((double) numBatches * ((double) miniBatchSize / (double) batchSize));

                    for (T_Size i = 0; i < predictedOutput.cols(); i++) {
                        int index1;
                        int index2;

                        predictedOutput.col(i).maxCoeff(&index1);
                        correctOutput.col(i).maxCoeff(&index2);

                        if (index1 == index2) {
                            accuracy++;
                        }
                    }
                }

                Impulse::NeuralNetwork::Trainer::CostGradientResult result;
                result.cost = cost;
                result.accuracy = (accuracy - 1) / (double) numberOfExamples * 100.0;
                if (rollGradient) {
                    result.gradient = this->network.getRolledGradient();
                }

                return result;
            }

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Adam>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Adadelta>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Adagrad>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::GradientDescent>::cost(Impulse::Dataset::SlicedDataset &dataSet,
                                                              bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Momentum>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Nesterov>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template
            Impulse::NeuralNetwork::Trainer::CostGradientResult
            AbstractTrainer<Optimizer::Rmsprop>::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient);

            template<typename OPTIMIZER_TYPE>
            void AbstractTrainer<OPTIMIZER_TYPE>::setStepCallback(std::function<void ()> callback) {
                this->stepCallback = callback;
                this->stepCallbackSet = true;
            }

            template
            void AbstractTrainer<Optimizer::Adam>::setStepCallback(std::function<void ()>);

            template
            void AbstractTrainer<Optimizer::Nesterov>::setStepCallback(std::function<void ()>);

            template
            void AbstractTrainer<Optimizer::Adadelta>::setStepCallback(std::function<void ()>);

            template
            void AbstractTrainer<Optimizer::Rmsprop>::setStepCallback(std::function<void ()>);

            template
            void AbstractTrainer<Optimizer::Momentum>::setStepCallback(std::function<void ()>);

            template
            void AbstractTrainer<Optimizer::GradientDescent>::setStepCallback(std::function<void ()>);

            template
            void AbstractTrainer<Optimizer::Adagrad>::setStepCallback(std::function<void ()>);
        }
    }
}