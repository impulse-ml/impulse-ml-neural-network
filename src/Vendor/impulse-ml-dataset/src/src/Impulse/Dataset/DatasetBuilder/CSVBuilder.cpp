#include "../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetBuilder {

            CSVBuilder::CSVBuilder(T_String path) {
                this->path = std::move(path);
            }

            CSVBuilder::~CSVBuilder() {
                this->closeFile();
            }

            void CSVBuilder::openFile() {
                this->fileHandle.open(this->path);
                if (!this->fileHandle.is_open()) {
                    std::cout << "Cannot open file." << std::endl;
                    throw "Cannot open file.";
                }
            }

            void CSVBuilder::closeFile() {
                if (this->fileHandle.is_open()) {
                    this->fileHandle.close();
                }
            }

            T_StringVector CSVBuilder::parseLine(T_String &line) {
                CSVState state = CSVState::UnquotedField;
                T_StringVector fields{""};
                T_Size i = 0; // index of the current field

                line.erase(std::find_if(line.rbegin(), line.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), line.end());

                for (char c : line) {
                    switch (state) {
                        case CSVState::UnquotedField:
                            switch (c) {
                                case ',': // end of field
                                    fields.emplace_back("");
                                    i++;
                                    break;
                                case '"':
                                    state = CSVState::QuotedField;
                                    break;
                                default:
                                    fields[i].push_back(c);
                                    break;
                            }
                            break;
                        case CSVState::QuotedField:
                            switch (c) {
                                case '"':
                                    state = CSVState::QuotedQuote;
                                    break;
                                default:
                                    fields[i].push_back(c);
                                    break;
                            }
                            break;
                        case CSVState::QuotedQuote:
                            switch (c) {
                                case ',': // , after closing quote
                                    fields.emplace_back("");
                                    i++;
                                    state = CSVState::UnquotedField;
                                    break;
                                case '"': // "" -> "
                                    fields[i].push_back('"');
                                    state = CSVState::QuotedField;
                                    break;
                                default:  // end of quote
                                    state = CSVState::UnquotedField;
                                    break;
                            }
                            break;
                    }
                }

                return fields;
            }

            Dataset CSVBuilder::build() {
                Dataset dataset;

                this->openFile();

                T_String line;
                while (std::getline(this->fileHandle, line)) {
                    T_StringVector fields = this->parseLine(line);
                    dataset.addSample(this->createSample(fields));
                }

                this->closeFile();

                return dataset;
            }
        }
    }
}
