#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            ClassifierBuilder::ClassifierBuilder(T_Dimension dims) : Abstract<Network::ClassifierNetwork>(dims) {
            }

            void ClassifierBuilder::firstLayerTransition(Layer::LayerPointer layer) {
                layer->setPrevSize(this->dimension.width);
            }
        }
    }
}
