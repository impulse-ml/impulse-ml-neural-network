#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Conv::Conv() : Abstract3D() {}

            void Conv::configure() {
                this->W.resize(this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->W.setRandom();
                this->W = this->W.unaryExpr([this](const double x) {
                    return x * sqrt(2.0 / (this->width * this->height * this->depth));
                });

                this->b.resize(this->numFilters, 1);
                this->b = this->b.unaryExpr([](const double x) {
                    return 0.01;
                });

                this->gW.resize(this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->gb.resize(this->numFilters, 1);
            }

            Eigen::MatrixXd Conv::forward(const Eigen::MatrixXd &input) {
                this->Z = input;

                Eigen::MatrixXd result(this->getOutputWidth() * this->getOutputHeight() * this->getOutputDepth(), input.cols());

#pragma omp parallel
#pragma omp for
                for (T_Size i = 0; i < input.cols(); i++) {
                    Eigen::MatrixXd conv = Utils::im2col(input.col(i), this->depth,
                                                        this->height, this->width,
                                                        this->filterSize, this->filterSize,
                                                        this->padding, this->padding,
                                                        this->stride, this->stride);

                    Eigen::MatrixXd tmp = ((this->W * conv).colwise() + this->b).transpose(); // transpose for
                    // rolling to vector
                    Eigen::Map<Eigen::VectorXd> tmp2(tmp.data(), tmp.size());
                    result.col(i) = tmp2;
                }

                this->A = this->activation(result);
                return this->A;
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

            Eigen::MatrixXd Conv::activation(Eigen::MatrixXd &m) {
                return m.unaryExpr([](const double x) {
                    return std::max(0.0, x); // TODO: set it; RELU by default
                });
            }

            Eigen::MatrixXd Conv::derivative() {
                return this->A.unaryExpr([](const double x) { // TODO: derivative for RELU
                    if (x > 0.0) {
                        return 1.0;
                    }
                    return 0.0;
                });
            }

            const T_String Conv::getType() {
                return TYPE_CONV;
            }

            double Conv::loss(Eigen::MatrixXd output, Eigen::MatrixXd predictions) {
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
