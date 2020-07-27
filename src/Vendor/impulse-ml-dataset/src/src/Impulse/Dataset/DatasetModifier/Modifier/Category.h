#pragma once

#include "../../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                class Category : public Abstract {
                public:
                    explicit Category(Dataset &dataset) : Abstract(dataset) {}

                    void applyToColumn(int columnIndex) override;
                };
            }
        }
    }
}
