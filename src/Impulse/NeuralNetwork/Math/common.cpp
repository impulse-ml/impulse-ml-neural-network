#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            Math::T_RawVector vectorToRaw(Math::T_Vector &vec) {
                return Math::T_RawVector(vec.data(), vec.data() + vec.rows() * vec.cols());
            }

            Math::T_Vector rawToVector(Math::T_RawVector &vec) {
                return Eigen::Map<Math::T_Vector, Eigen::Unaligned>(vec.data(), vec.size());
            }
        }
    }
}
