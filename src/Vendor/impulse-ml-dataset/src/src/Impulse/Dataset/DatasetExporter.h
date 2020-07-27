#pragma once

#include "include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        class DatasetExporter {
        protected:
            Dataset &dataset;
        public:
            explicit DatasetExporter(Dataset &dataset);

            void exportToFile(T_String);
        };
    }
}
