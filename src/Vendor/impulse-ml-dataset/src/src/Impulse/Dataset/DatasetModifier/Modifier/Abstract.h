#pragma once

#include "../../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                class Abstract {
                protected:
                    Dataset &dataset;
                    std::map<T_String, std::map<double, T_String>> data;

                public:
                    explicit Abstract(Dataset &dataset);

                    virtual ~Abstract() = default;

                    void apply();

                    virtual void applyToColumn(int columnIndex) = 0;
                };
            }
        }
    }
}
