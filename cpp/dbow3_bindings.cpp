#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <DBoW3.h>
#include <opencv2/core.hpp>

#include <memory>
#include <stdexcept>
#include <vector>


namespace py = pybind11;


cv::Mat numpy_to_cv_mat(
    const py::array_t<unsigned char, py::array::c_style | py::array::forcecast>& array
)
{
    py::buffer_info info = array.request();

    if (info.ndim != 2)
    {
        throw std::runtime_error(
            "ORB descriptors must be a 2D uint8 NumPy array."
        );
    }

    if (info.shape[1] != 32)
    {
        throw std::runtime_error(
            "ORB descriptors must have shape (N, 32)."
        );
    }

    if (info.shape[0] == 0)
    {
        return cv::Mat();
    }

    cv::Mat descriptors(
        static_cast<int>(info.shape[0]),
        static_cast<int>(info.shape[1]),
        CV_8U,
        info.ptr
    );

    return descriptors.clone();
}


class PyVocabulary
{
public:

    PyVocabulary()
        : vocabulary_(
            std::make_shared<DBoW3::Vocabulary>()
        )
    {
    }


    bool load(
        const std::string& path
    )
    {
        vocabulary_->load(
            path
        );
    
        return !vocabulary_->empty();
    }


    void save(
        const std::string& path
    ) const
    {
        vocabulary_->save(
            path
        );
    }


    void create(
        const std::vector<
            py::array_t<
                unsigned char,
                py::array::c_style | py::array::forcecast
            >
        >& descriptor_sets,
        int branching_factor=10,
        int depth=5
    )
    {
        if (descriptor_sets.empty())
        {
            throw std::runtime_error(
                "descriptor_sets cannot be empty."
            );
        }

        std::vector<cv::Mat> training_features;

        training_features.reserve(
            descriptor_sets.size()
        );

        for (const auto& descriptors : descriptor_sets)
        {
            cv::Mat mat = numpy_to_cv_mat(
                descriptors
            );

            if (mat.empty())
            {
                continue;
            }

            training_features.push_back(
                mat
            );
        }

        if (training_features.empty())
        {
            throw std::runtime_error(
                "No valid descriptor sets were provided."
            );
        }

        vocabulary_ = std::make_shared<DBoW3::Vocabulary>(
            branching_factor,
            depth,
            DBoW3::TF_IDF,
            DBoW3::L1_NORM
        );

        vocabulary_->create(
            training_features
        );
    }


    std::size_t size() const
    {
        return vocabulary_->size();
    }


    double score(
        const py::array_t<
            unsigned char,
            py::array::c_style | py::array::forcecast
        >& descriptors_a,
        const py::array_t<
            unsigned char,
            py::array::c_style | py::array::forcecast
        >& descriptors_b
    ) const
    {
        cv::Mat mat_a = numpy_to_cv_mat(
            descriptors_a
        );

        cv::Mat mat_b = numpy_to_cv_mat(
            descriptors_b
        );

        if (mat_a.empty() || mat_b.empty())
        {
            return 0.0;
        }

        DBoW3::BowVector bow_a;
        DBoW3::BowVector bow_b;

        vocabulary_->transform(
            mat_a,
            bow_a
        );

        vocabulary_->transform(
            mat_b,
            bow_b
        );

        return vocabulary_->score(
            bow_a,
            bow_b
        );
    }


    std::shared_ptr<DBoW3::Vocabulary> get_vocabulary() const
    {
        return vocabulary_;
    }


private:

    std::shared_ptr<DBoW3::Vocabulary> vocabulary_;
};


class PyDatabase
{
public:

    PyDatabase(
        const PyVocabulary& vocabulary
    )
        : vocabulary_(
            vocabulary.get_vocabulary()
        ),
          database_(
            *vocabulary_,
            false,
            0
        )
    {
        if (!vocabulary_ || vocabulary_->empty())
        {
            throw std::runtime_error(
                "Vocabulary must be loaded or created before creating the database."
            );
        }
    }


    int add(
        const py::array_t<
            unsigned char,
            py::array::c_style | py::array::forcecast
        >& descriptors
    )
    {
        cv::Mat mat = numpy_to_cv_mat(
            descriptors
        );

        if (mat.empty())
        {
            throw std::runtime_error(
                "Cannot add empty descriptors to the database."
            );
        }

        return static_cast<int>(
            database_.add(
                mat
            )
        );
    }


    std::vector<std::pair<int, double>> query(
        const py::array_t<
            unsigned char,
            py::array::c_style | py::array::forcecast
        >& descriptors,
        int max_results=10
    ) const
    {
        if (max_results <= 0)
        {
            return {};
        }

        cv::Mat mat = numpy_to_cv_mat(
            descriptors
        );

        if (mat.empty())
        {
            return {};
        }

        DBoW3::QueryResults results;

        database_.query(
            mat,
            results,
            max_results
        );

        std::vector<std::pair<int, double>> output;

        output.reserve(
            results.size()
        );

        for (const auto& result : results)
        {
            output.emplace_back(
                static_cast<int>(result.Id),
                static_cast<double>(result.Score)
            );
        }

        return output;
    }


    std::size_t size() const
    {
        return database_.size();
    }


private:

    std::shared_ptr<DBoW3::Vocabulary> vocabulary_;
    DBoW3::Database database_;
};


PYBIND11_MODULE(dbow3_bindings, module)
{
    module.doc() =
        "Minimal DBoW3 bindings for ORB-based SLAM loop detection";


    py::class_<PyVocabulary>(
        module,
        "Vocabulary"
    )
        .def(
            py::init<>()
        )
        .def(
            "load",
            &PyVocabulary::load,
            py::arg("path")
        )
        .def(
            "save",
            &PyVocabulary::save,
            py::arg("path")
        )
        .def(
            "create",
            &PyVocabulary::create,
            py::arg("descriptor_sets"),
            py::arg("branching_factor") = 10,
            py::arg("depth") = 5
        )
        .def(
            "size",
            &PyVocabulary::size
        )
        .def(
            "score",
            &PyVocabulary::score,
            py::arg("descriptors_a"),
            py::arg("descriptors_b")
        );


    py::class_<PyDatabase>(
        module,
        "Database"
    )
        .def(
            py::init<const PyVocabulary&>(),
            py::arg("vocabulary")
        )
        .def(
            "add",
            &PyDatabase::add,
            py::arg("descriptors")
        )
        .def(
            "query",
            &PyDatabase::query,
            py::arg("descriptors"),
            py::arg("max_results") = 10
        )
        .def(
            "size",
            &PyDatabase::size
        );
}