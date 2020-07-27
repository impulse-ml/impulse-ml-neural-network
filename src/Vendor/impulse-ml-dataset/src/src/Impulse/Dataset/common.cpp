#include "include.h"

namespace Impulse {

    namespace Dataset {

        T_String toString(double value) {
            return std::to_string(value);
        }

        double toDouble(T_String value) {
            if (value.length() == 0) {
                return NAN;
            }
            return std::stod(value);
        }

        double MIN(double value1, double value2) {
            return std::min(value1, value2);
        }

        double MAX(double value1, double value2) {
            return std::max(value1, value2);
        }
    }
}
