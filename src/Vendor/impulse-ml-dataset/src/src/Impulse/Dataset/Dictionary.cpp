#include "include.h"

namespace Impulse {

    namespace Dataset {
        void replaceAll(std::string &str, const std::string &from, const std::string &to) {
            if (from.empty())
                return;
            size_t start_pos = 0;
            while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
                str.replace(start_pos, from.length(), to);
                start_pos += to.length();
            }
        }

        std::vector<T_String> Dictionary::loadWords(T_String filePath) {
            std::vector<T_String> result;

            std::ifstream infile(filePath);
            std::string line;
            while (std::getline(infile, line)) {
                std::istringstream iss(line);
                std::string s;
                while (getline(iss, s, ' ')) {
                    for (auto &c : s) {
                        c = tolower(c);
                    }
                    result.push_back(s);
                }
            }
            infile.close();

            return result;
        }

        std::vector<T_String> Dictionary::prepareWord(T_String word) {
            std::vector<T_String> result;
            T_String pushed_word = "";

            for (T_Size i = 0; i < word.length(); i += 1) {
                char ch = word.at(i);
                if (ch == '!' ||
                    ch == '.' ||
                    ch == ',' ||
                    ch == '?' ||
                    ch == '\\' ||
                    ch == '\'' ||
                    ch == '"' ||
                    ch == '(' ||
                    ch == ')' ||
                    ch == '-') {
                    if (pushed_word.length() > 0) {
                        result.push_back(pushed_word);
                        pushed_word = "";

                        T_String chStr = "";
                        chStr += ch;
                        result.push_back(chStr);
                    } else {
                        pushed_word += ch;
                    }
                } else {
                    pushed_word += ch;
                }
                if (i == (word.length() - 1) && pushed_word.length() > 0) {
                    result.push_back(pushed_word);
                    pushed_word = "";
                }
            }

            return result;
        }

        std::vector<Eigen::VectorXd> Dictionary::loadWordsMultidimentional(std::vector<T_String> dictionary, T_String filePath) {
            std::vector<Eigen::VectorXd> result;

            std::ifstream infile(filePath);
            std::string line;
            while (std::getline(infile, line)) {
                std::istringstream iss(line);
                std::string word;
                std::vector<T_String> words;
                while (getline(iss, word, ' ')) {
                    for (auto &c : word) {
                        c = tolower(c);
                    }
                    std::vector<T_String> newWords = Dictionary::prepareWord(word);
                    words.insert(words.end(), newWords.begin(), newWords.end());
                }
                for (T_Size i = 0; i < words.size(); i += 1) {
                    Eigen::VectorXd seq;
                    seq.resize(dictionary.size());
                    for (T_Size j = 0; j < dictionary.size(); j += 1) {
                        seq(j) = (dictionary.at(j).compare(words.at(i)) == 0) ? 1 : 0;
                    }
                    result.push_back(seq);
                }
            }
            infile.close();

            return result;
        }

        std::vector<T_String> Dictionary::prepareWords(std::vector<T_String> words) {
            std::sort(words.begin(), words.end());
            words.erase(std::unique(words.begin(), words.end()), words.end());

            std::vector<T_String> newWords;

            std::vector<T_String> specialCharacters = {"!", ".", ",", "?", "\"", "'", "(", ")", "-"};
            std::string pushed_word = "";
            for (auto &word : words) {
                for (T_Size i = 0; i < word.length(); i += 1) {
                    char ch = word.at(i);
                    if (ch == '!' ||
                        ch == '.' ||
                        ch == ',' ||
                        ch == '?' ||
                        ch == '\\' ||
                        ch == '\'' ||
                        ch == '"' ||
                        ch == '(' ||
                        ch == ')' ||
                        ch == '-') {
                        if (pushed_word.length() > 0) {
                            newWords.push_back(pushed_word);
                            pushed_word = "";

                            T_String chStr = "";
                            chStr += ch;
                            newWords.push_back(chStr);
                        } else {
                            pushed_word += ch;
                        }
                    } else {
                        pushed_word += ch;
                    }
                    if (i == (word.length() - 1) && pushed_word.length() > 0) {
                        newWords.push_back(pushed_word);
                        pushed_word = "";
                    }
                }
            }

            std::sort(newWords.begin(), newWords.end());
            newWords.erase(std::unique(newWords.begin(), newWords.end()), newWords.end());

            return newWords;
        }

        void Dictionary::makeDictionary(T_String input, T_String output, T_String result) {
            std::vector<T_String> inputWords = Dictionary::loadWords(input);
            std::vector<T_String> outputWords = Dictionary::loadWords(output);

            inputWords.insert(inputWords.end(), outputWords.begin(), outputWords.end());

            std::vector<T_String> words = Dictionary::prepareWords(inputWords);

            std::ofstream resultFile;
            resultFile.open(result);
            for (T_Size i = 0; i < words.size(); i += 1) {
                resultFile << words.at(i) << "\n";
            }
            resultFile.close();
        }

        Dictionary Dictionary::load(T_String dictionary, T_String input, T_String output) {
            std::vector<T_String> words;

            std::ifstream infile(dictionary);
            std::string line;
            while (std::getline(infile, line)) {
                words.push_back(line);
            }

            auto instance = Dictionary();

            instance.dictionary = words;
            instance.input = Dictionary::loadWordsMultidimentional(words, input);
            instance.output = Dictionary::loadWordsMultidimentional(words, output);

            return instance;
        }
    }

    std::vector<Eigen::VectorXd> Dictionary::getInput() {
        return this->input;
    }

    std::vector<Eigen::VectorXd> Dictionary::getOutput() {
        return this->output;
    }
}

