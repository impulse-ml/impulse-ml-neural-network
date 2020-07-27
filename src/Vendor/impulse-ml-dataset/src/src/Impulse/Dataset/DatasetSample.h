#pragma once

#include "include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        class DatasetSample {
        protected:
            T_StringVector rawData;
        public:
            explicit DatasetSample(T_StringVector vec);

            explicit DatasetSample(T_StringVector *vec);

            explicit DatasetSample(T_DoubleVector vec);

            DatasetSample(std::initializer_list<double> list);

            void out();

            T_String getColumnToString(int columnIndex);

            double getColumnToDouble(int columnIndex);

            void setColumn(int columnIndex, T_String value);

            void setColumn(int columnIndex, double value);

            void setColumn(int columnIndex, std::function<T_String(T_String)> callback);

            void insertColumnAfter(int columnIndex, T_String value);

            void removeColumnAt(int columnIndex);

            T_Size getSize();

            Eigen::VectorXd exportToEigen();
        };
    }
}
