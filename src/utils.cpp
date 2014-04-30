#include "utils.h"

#include <sstream>
#include <algorithm>
#include <functional>
#include <locale>
#include <set>

std::vector<std::string> Utils::split(const std::string & text, char delim) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string item;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

std::string& Utils::ltrim(std::string & text) {
    text.erase(text.begin(), find_if(text.begin(), text.end(), not1(std::ptr_fun<int, int>(isspace))));
    return text;
}

std::string& Utils::rtrim(std::string & text) {
    text.erase(find_if(text.rbegin(), text.rend(), not1(std::ptr_fun<int, int>(isspace))).base(), text.end());
    return text;
}

std::string& Utils::trim(std::string & text) {
    return ltrim(rtrim(text));
}

bool Utils::exists(const std::string & filename) {
    if (FILE *file = fopen(filename.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    } 
}
