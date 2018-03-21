#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <list>
#include <algorithm> 


const unsigned int INSERT_COST = 1;
const unsigned int DELETE_COST = 1;
const unsigned int REPLACE_COST = 1;


unsigned int min(unsigned int a, unsigned int b, unsigned int c) {
    if ((a < b) && (a < c)) {
        return a;
    } else if ((b < a) && (b < c)) {
        return b;
    } else {
        return c;
    }
}


unsigned int levenshtein_distance(const std::string& first_str, const std::string& second_str) {
    unsigned int d[first_str.size()][second_str.size()];
    d[0][0] = 0;
    for (int j = 1; j < second_str.size(); ++j) {
        d[0][j] = d[0][j - 1] + INSERT_COST;
    }
    for (int i = 1; i < first_str.size(); ++i) {
        d[i][0] = d[i - 1][0] + DELETE_COST;
        for (int j = 1; j < second_str.size(); ++j) {
            d[i][j] = min(
                d[i - 1][j] + DELETE_COST,
                d[i][j - 1] + INSERT_COST,
                d[i - 1][j - 1] + ((first_str[i] != second_str[j]) ? REPLACE_COST : 0)
            );
        }
    }
    return d[first_str.size() - 1][second_str.size() - 1];
}


bool load_data(std::vector<std::string>& data, const std::vector<std::string>& file_names) {
    for (int i = 0; i < file_names.size(); ++i) {
        std::ifstream file(file_names[i]);
        if (!file.is_open()) {
            return false;
        }
        for (std::string line; std::getline(file, line);) {
            data.push_back(line);
        }
    }

    // leave only unique elements
    std::sort(data.begin(), data.end());
    auto last = std::unique(data.begin(), data.end());
    data.erase(last, data.end()); 

    return true;
}


bool write_nearest(std::ofstream& table_file,
                 const std::vector<std::string>& data,
                 const std::string& word,
                 unsigned int n_nearest) {
    std::list<std::string> nearest = {data[0]};
    for (int i = 1; i < data.size(); ++i) {
        if (
            (nearest.size() == n_nearest) &&
            (levenshtein_distance(word, data[i]) >= levenshtein_distance(word, nearest.front()))
        ) {
            continue;
        }
        bool inserted = false;
        for (auto it = nearest.begin(); it != nearest.end(); ++it) {
            if (levenshtein_distance(data[i], word) >= levenshtein_distance(*it, word)) {
                nearest.insert(it, data[i]);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            nearest.push_back(data[i]);
        }
        if (nearest.size() > n_nearest) {
            nearest.pop_front();
        }
    }
    for (auto it = nearest.rbegin(); it != nearest.rend(); ++it) {
        table_file << *it << " ";
    }
    table_file << "\n";
}


bool make_table(const std::vector<std::string>& data, const std::string& table_file_name, unsigned int n_nearest) {
    std::ofstream table_file(table_file_name);
    if (!table_file.is_open()) {
        return false;
    }

    for (int i = 0; i < data.size(); ++i) {
        write_nearest(table_file, data, data[i], n_nearest);
        if (i % 10 == 0) {
            std::cout << float(i) * 100 / data.size() << " %" << std::endl;
        }
    }
    return true;
}


int main(int argc, char** argv) {
    // argument 1: str - path where to read the data
    // argument 2: str - output filename
    // argument 3: uint - number of nearest words to find
    if (argc < 4) {
        std::cout << "Wrong arguments number!" << std::endl;
        return 0;
    }
    std::string read_path(argv[1]);
    std::string table_file_name(argv[2]);
    std::vector<std::string> file_names = {
        read_path + "/he.test.txt",
        read_path + "/he.train.txt",
        read_path + "/he.dev.txt"
    };
    std::vector<std::string> data;
    if (load_data(data, file_names)) {
        make_table(data, table_file_name, std::stoi(argv[3]));
    } else {
        std::cout << "Error while loading data!" << std::endl;
    }
    return 0;
}
