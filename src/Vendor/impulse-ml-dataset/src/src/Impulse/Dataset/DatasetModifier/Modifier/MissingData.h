#pragma once

#include "../../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                class MissingData : public Abstract {
                protected:
                    T_String modificationType = "mean";

                public:
                    explicit MissingData(Dataset &dataset) : Abstract(dataset) {}

                    void setModificationType(T_String type);

                    void applyToColumn(int columnIndex) override;
                };
            }
        }
    }
}
