#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            Math::T_RawVector vectorToRaw(Eigen::VectorXd &vec) {
                return Math::T_RawVector(vec.data(), vec.data() + vec.rows() * vec.cols());
            }

            Eigen::VectorXd rawToVector(Math::T_RawVector &vec) {
                return Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vec.data(), vec.size());
            }
        }
    }
}
