#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Conv::Conv() : Abstract3D() {}

            void Conv::configure() {
                // initialize weights
                this->computation->resize("W", this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->computation->randomInit("W", this->width * this->height * this->depth);

                // initialize bias
                this->computation->resize("b", this->numFilters, 1);
                this->computation->randomInit("b", 0.01);

                // initialize gradient
                this->computation->resize("gW", this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->computation->setZero("gW");

                this->computation->resize("gB", this->numFilters, 1);
                this->computation->setZero("gB");

                // initialize optimizer variables
                this->computation->resize("cW", this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->computation->setZero("cW");

                this->computation->resize("cB", this->numFilters, 1);
                this->computation->setZero("cB");

                this->computation->resize("vW", this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->computation->setZero("vW");

                this->computation->resize("vB", this->numFilters, 1);
                this->computation->setZero("vB");

                this->computation->resize("wW", this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->computation->setZero("wW");

                this->computation->resize("wB", this->numFilters, 1);
                this->computation->setZero("wB");
            }

            Eigen::MatrixXd Conv::forward(const Eigen::MatrixXd &input) {
                Eigen::MatrixXd result(this->getOutputWidth() * this->getOutputHeight() * this->getOutputDepth(),
                                       input.cols());

#pragma omp parallel
                for (T_Size i = 0; i < input.cols(); i++) {
                    Eigen::MatrixXd conv = Utils::im2col(input.col(i), this->depth,
                                                         this->height, this->width,
                                                         this->filterSize, this->filterSize,
                                                         this->padding, this->padding,
                                                         this->stride, this->stride);

                    Eigen::MatrixXd tmp = this->computation->forward(conv).transpose(); // transpose for
                    // rolling to vector
                    Eigen::Map<Eigen::VectorXd> tmp2(tmp.data(), tmp.size());
                    result.col(i) = tmp2;
                }

                this->computation->setVariable("Z", result);
                this->activation();
                //this->computation->setVariable("Z", input);
                return this->computation->getVariable("A");
            }

            T_Size Conv::getOutputHeight() {
                return (this->width - this->filterSize + 2 * this->padding) / this->stride + 1;
            }

            T_Size Conv::getOutputWidth() {
                return (this->height - this->filterSize + 2 * this->padding) / this->stride + 1;
            }

            T_Size Conv::getOutputDepth() {
                return this->numFilters;
            }

            void Conv::setFilterSize(T_Size value) {
                this->filterSize = value;
            }

            T_Size Conv::getFilterSize() {
                return this->filterSize;
            }

            void Conv::setPadding(T_Size value) {
                this->padding = value;
            }

            T_Size Conv::getPadding() {
                return this->padding;
            }

            void Conv::setStride(T_Size value) {
                this->stride = value;
            }

            T_Size Conv::getStride() {
                return this->stride;
            }

            void Conv::setNumFilters(T_Size value) {
                this->numFilters = value;
            }

            T_Size Conv::getNumFilters() {
                return this->numFilters;
            }

            Eigen::MatrixXd Conv::activation() {
                this->computation->reluActivation();
                return this->computation->getVariable("A");
            }

            Eigen::MatrixXd Conv::derivative(Eigen::MatrixXd &a) {
                return this->computation->reluDerivative(a);
            }

            const T_String Conv::getType() {
                return TYPE_CONV;
            }

            double Conv::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                static_assert("No loss for CONV layer.", "");
                return 0.0;
            }

            double Conv::error(T_Size m) {
                static_assert("No error for CONV layer.", "");
                return 0.0;
            }
        }
    }
}
