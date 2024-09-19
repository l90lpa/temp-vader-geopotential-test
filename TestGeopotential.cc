#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <eckit/config/YAMLConfiguration.h>

#include "vader/recipes/Geopotential.h"

using json = nlohmann::json;

// ============================================================================

namespace std {
    template<typename T>
    std::string to_string(const std::vector<T>& v) {
        std::string str = "[";
        for (const auto& e : v) {
            str += std::to_string(e) + ", ";
        }
        if (!v.empty()) {
            str.pop_back(); // remove last space
            str.pop_back(); // remove last comma
        }
        str += "]";
        return str;
    }
}


std::tuple<atlas::Field, size_t, size_t, size_t> vectorsToField(const std::string& name, const std::vector<std::vector<std::vector<double>>>& v) {
    size_t lvls = v.size();
    size_t lons = v[0].size();
    size_t lats = v[0][0].size();
    atlas::Field f(name, atlas::array::make_datatype<double>(), atlas::array::make_shape(lons * lats, lvls));
    auto f_view = atlas::array::make_view<double, 2>(f);
    for (size_t i = 0; i < lvls; ++i) {
        for (size_t j = 0; j < lons; ++j) {
            for (size_t k = 0; k < lats; ++k) {
                f_view(j * lats + k, i) = v[i][j][k];
            }
        }
    }
    return std::make_tuple(f, lvls, lons, lats);
}


std::vector<std::vector<std::vector<double>>> fieldToVectors(atlas::Field& f, size_t lvls, size_t lons, size_t lats) {
    std::vector<std::vector<std::vector<double>>> v(lvls, std::vector<std::vector<double>>(lons, std::vector<double>(lats)));
    auto f_view = atlas::array::make_view<double, 2>(f);
    for (size_t i = 0; i < lvls; ++i) {
        for (size_t j = 0; j < lons; ++j) {
            for (size_t k = 0; k < lats; ++k) {
                v[i][j][k] = f_view(j * lats + k, i);
            }
        }
    }
    return v;
}


std::tuple<atlas::Field, size_t, size_t> vectorsToField(const std::string& name, const std::vector<std::vector<double>>& v) {
    size_t lons = v.size();
    size_t lats = v[0].size();
    atlas::Field f(name, atlas::array::make_datatype<double>(), atlas::array::make_shape(lons * lats));
    auto f_view = atlas::array::make_view<double, 1>(f);
    for (size_t i = 0; i < lons; ++i) {
        for (size_t j = 0; j < lats; ++j) {
            f_view(i * lats + j) = v[i][j];
        }
    }
    return std::make_tuple(f, lons, lats);
}


std::vector<std::vector<double>> fieldToVectors(atlas::Field& f, size_t lons, size_t lats) {
    std::vector<std::vector<double>> v(lons, std::vector<double>(lats));
    auto f_view = atlas::array::make_view<double, 1>(f);
    for (size_t i = 0; i < lons; ++i) {
        for (size_t j = 0; j < lats; ++j) {
            v[i][j] = f_view(i * lats + j);
        }
    }
    return v;
}

// ============================================================================


int main() {

    std::ifstream ifs("./bin/diagnostic_data.json");
    json jf = json::parse(ifs);

    // print top level keys
    for (auto& [key, val] : jf.items()) {
        std::cout << "key: " << key << '\n';
    }

    // get raw data from json
    std::vector<std::vector<std::vector<double>>> raw_temperature;
    std::vector<std::vector<std::vector<double>>> raw_specificHumidity;
    std::vector<std::vector<double>> raw_nodalOrography;
    std::vector<double> raw_sigmaCoords;
    jf.at("nodal_orography").get_to(raw_nodalOrography);
    jf.at("temperature").get_to(raw_temperature);
    jf.at("specific_humidity").get_to(raw_specificHumidity);
    jf.at("coordinates").at("centers").get_to(raw_sigmaCoords);

    // convert to atlas fields
    auto [temperature, temperature_lvls, temperature_lons, temperature_lats] = vectorsToField("temperature", raw_temperature);
    auto [specificHumidity, specificHumidity_lvls, specificHumidity_lons, specificHumidity_lats] = vectorsToField("specific_humidity", raw_specificHumidity);
    auto [nodalOrography, nodalOrography_lons, nodalOrography_lats] = vectorsToField("orography", raw_nodalOrography);

    // make sigma coords field
    atlas::Field sigmaCoords("sigma_coords", atlas::array::make_datatype<double>(), temperature.shape());
    {
        auto sigmaCoords_view = atlas::array::make_view<double, 2>(sigmaCoords);
        for(size_t i = 0; i < sigmaCoords_view.shape(0); ++i) {
            for (size_t j = 0; j < sigmaCoords_view.shape(1); ++j) {
                sigmaCoords_view(i, j) = raw_sigmaCoords[j];
            }
        }
    }

    // make input fieldset
    atlas::FieldSet fs;
    fs.add(atlas::Field("geopotential", atlas::array::make_datatype<double>(), temperature.shape()));
    fs.add(temperature);
    fs.add(specificHumidity);
    fs.add(nodalOrography);
    fs.add(sigmaCoords);

    // make params and get config
    vader::Geopotential_Parameters params;
    eckit::YAMLConfiguration config("water_vapor_gas_constant: 0.0005339522375674384\nideal_gas_constant: 0.00033225165572835946\ngravity_acceleration: 72.36408283456718");

    // make recipe object
    vader::Geopotential geopotentialRecipe(params, config);

    // // execute recipe
    geopotentialRecipe.executeNL(fs);

    {
        std::vector<std::vector<std::vector<double>>> raw_test_geopotential;
        jf.at("outputs").get_to(raw_test_geopotential);
        auto [test_geopotential, lvls, lons, lats] = vectorsToField("test_geopotential", raw_test_geopotential);

        // find the max absolute difference between geopotential and test_geopotential
        auto geopotential = fs.field("geopotential");
        auto geopotential_view = atlas::array::make_view<double, 2>(geopotential);
        auto test_geopotential_view = atlas::array::make_view<double, 2>(test_geopotential);
        double max_diff = 0.0;
        for (size_t i = 0; i < geopotential_view.shape(0); ++i) {
            for (size_t j = 0; j < geopotential_view.shape(1); ++j) {
                max_diff = std::max(max_diff, std::abs(geopotential_view(i, j) - test_geopotential_view(i, j)));
            }
        }
        std::cout << "(geopotential-other) Max absolute difference: " << max_diff << std::endl;
    }

    return 0;
}
