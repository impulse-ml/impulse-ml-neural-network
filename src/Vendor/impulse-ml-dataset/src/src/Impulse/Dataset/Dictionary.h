#pragma once

#include "include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        void replaceAll(std::string &, const std::string &, const std::string &);

        class Dictionary {
        protected:
            std::vector<T_String> dictionary;
            std::vector<Eigen::VectorXd> input;
            std::vector<Eigen::VectorXd> output;
        public:
            void static makeDictionary(T_String, T_String, T_String);
            std::vector<T_String> static loadWords(T_String);
            std::vector<T_String> static prepareWords(std::vector<T_String>);
            std::vector<Eigen::VectorXd> static loadWordsMultidimentional(std::vector<T_String>, T_String);
            std::vector<T_String> static prepareWord(T_String);
            std::vector<Eigen::VectorXd> getInput();
            std::vector<Eigen::VectorXd> getOutput();

            static Dictionary load(T_String, T_String, T_String);
        };
    }
}
