#pragma once

#include "include.h"

namespace Impulse {

    namespace Dataset {

        typedef std::string T_String;
        typedef std::vector<T_String> T_StringVector;
        typedef std::vector<double> T_DoubleVector;
        typedef std::vector<int> T_IntVector;
        typedef unsigned int T_Size;

        T_String toString(double value);

        double toDouble(T_String value);

        double MIN(double value1, double value2);

        double MAX(double value1, double value2);
    }
}
