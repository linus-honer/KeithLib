#include "TensorImpl.h"

#include <memory>
#include <cmath>
#include <iomanip>
#include <random>
#include <ctime>

namespace keith {

    TensorImpl::TensorImpl(const Storage& storage, const Shape& shape, const Array<index_t>& stride) :
        _storage(storage), _shape(shape), _stride(stride) {}
    TensorImpl::TensorImpl(const Storage& storage, const Shape& shape) :
        _storage(storage), _shape(shape), _stride(shape.n_dim()) {
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim() - 1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i + 1);
            if (shape[i] == 1) _stride[i] = 0;
        }
    }
    TensorImpl::TensorImpl(const Shape& shape) :
        _storage(shape.d_size()), _shape(shape), _stride(shape.n_dim()) {
        for (int i = 0; i < shape.d_size(); ++i)
            _storage[i] = 0;
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim() - 1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i + 1);
            if (shape[i] == 1) _stride[i] = 0;
        }
    }
    TensorImpl::TensorImpl(const data_t* data, const Shape& shape) :
        _storage(shape.d_size()), _shape(shape), _stride(shape.n_dim()) {
        for (int i = 0; i < shape.d_size(); ++i)
            _storage[i] = data[i];
        for (int i = 0; i < shape.n_dim(); ++i) {
            if (i == shape.n_dim() - 1) _stride[i] = 1;
            else _stride[i] = shape.sub_size(i + 1);
            if (shape[i] == 1) _stride[i] = 0;
        }
    }
    TensorImpl::TensorImpl(Storage&& storage, Shape&& shape, Array<index_t>&& stride) :
        _storage(std::move(storage)), _shape(std::move(shape)), _stride(std::move(stride)) {}



    bool TensorImpl::is_contiguous() const
    {
        for (int i = 0; i < n_dim() - 1; ++i) {
            if (_shape[i] == 1) continue;
            if (_stride[i] != _shape.sub_size(i + 1)) return false;
        }
        if (_stride[n_dim() - 1] != 1) return false;
        return true;
    }

    data_t& TensorImpl::operator[](std::initializer_list<index_t> dims) {
        CHECK_EQUAL(n_dim(), dims.size(),
            "Invalid %zuD indices for %dD tensor", dims.size(), n_dim());
        index_t index = 0, dim = 0;
        for (auto v : dims) {
            CHECK_IN_RANGE(v, 0, size(dim),
                "Index out of range (expected to be in range of [0, %d), but got %d)",
                size(dim), v);
            index += v * _stride[dim];
            ++dim;
        }
        return _storage[index];
    }
    data_t TensorImpl::operator[](std::initializer_list<index_t> dims) const {
        CHECK_EQUAL(n_dim(), dims.size(),
            "Invalid %zuD indices for %dD tensor", dims.size(), n_dim());
        index_t index = 0, dim = 0;
        for (auto v : dims) {
            std::cout << v << " ";
            CHECK_IN_RANGE(v, 0, size(dim),
                "Index out of range (expected to be in range of [0, %d), but got %d)",
                size(dim), v);
            index += v * _stride[dim];
            ++dim;
        }
        std::cout << std::endl;
        return _storage[index];
    }

    data_t TensorImpl::item() const {
        CHECK_TRUE(n_dim() == 1 && size(0) == 1,
            "Only one element tensors can be converted to scalars");
        return _storage[0];
    }

    data_t TensorImpl::item(index_t idx) const {
        return _storage[idx];
    }

    data_t& TensorImpl::item(index_t idx)
    {
        return _storage[idx];
    }
    data_t TensorImpl::eval(Array<index_t> idx) const {
        int index = 0;
        if (idx.size() >= _shape.n_dim()) {
            for (int i = idx.size() - n_dim(); i < idx.size(); ++i)
                index += idx[i] * _stride[i - (idx.size() - n_dim())];
        }
        else {
            for (int i = 0; i < idx.size(); ++i)
                index += idx[i] * _stride[i + (n_dim() - idx.size())];
        }
        return item(index);
    }






    std::ostream& operator<<(std::ostream& out, const TensorImpl& tensor) {
        int max_width = 0;
        for (int i = 0; i < tensor.d_size(); ++i) {
            int value = (int)std::abs(tensor.item(i));
            int dig = value = (int)(std::log10(value)) + 1;
            if (tensor.item(i) < 0) ++dig;
            max_width = std::max(max_width, dig);
        }
        int cnt = 0, idx = 0, end_flag = tensor.n_dim();
        std::vector<int> dim_cnt(tensor.n_dim());
        while (cnt < tensor.d_size()) {
            for (int i = 0; i < tensor.n_dim() - end_flag; ++i)
                out << " ";
            for (int i = 0; i < end_flag; ++i)
                out << "[";
            out << std::setw(max_width + 4 + 1) << std::right << std::setprecision(4) << std::fixed;
            out << tensor.item(idx);
            end_flag = 0;
            for (int i = (int)tensor.n_dim() - 1; i >= 0; --i) {
                if (dim_cnt[i] + 1 < tensor.size()[i]) {
                    idx += tensor.stride()[i];
                    ++dim_cnt[i];
                    break;
                }
                else {
                    idx -= ((int)tensor.size()[i] - 1) * tensor.stride()[i];
                    dim_cnt[i] = 0;
                    ++end_flag;
                }
            }
            if (end_flag == 0) out << ", ";
            else {
                for (int i = 0; i < end_flag; ++i) {
                    out << "]";
                }
                out << std::endl;

            }
            ++cnt;
        }
        return out;
    }

    data_t TensorImpl::sum() const {
        data_t res = 0;
        std::vector<index_t> idx(n_dim(), 0);
        for (int i = 0; i < d_size(); ++i) {
            int cnt = 0;
            res += eval(idx);
            for (int j = 0; j < n_dim(); ++j) {
                if (idx[j] + 1 < size()[j]) {
                    ++idx[j];
                    break;
                }
                else {
                    idx[j] = 0;
                    ++cnt;
                }
            }
            if (cnt == n_dim()) break;
        }
        return res;
    }

    TensorImpl TensorMaker::ones(const Shape& shape) {
        TensorImpl tensor(shape);
        for (int i = 0; i < tensor.d_size(); ++i)
            tensor.item(i) = 1;
        return tensor;
    }

    TensorImpl TensorMaker::ones_like(const TensorImpl& tensor) {
        return ones(tensor.size());
    }

    TensorImpl TensorMaker::zeros(const Shape& shape) {
        TensorImpl tensor(shape);
        for (int i = 0; i < tensor.d_size(); ++i)
            tensor.item(i) = 0;
        return tensor;
    }

    TensorImpl TensorMaker::zeros_like(const TensorImpl& tensor) {
        return zeros(tensor.size());
    }

    TensorImpl TensorMaker::rand(const Shape& shape) {
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_real_distribution<data_t> dis(0, 1);
        TensorImpl tensor(shape);
        for (int i = 0; i < tensor.d_size(); ++i)
            tensor.item(i) = dis(gen);
        return tensor;
    }

    TensorImpl TensorMaker::rand_like(const TensorImpl& tensor) {
        return rand(tensor.size());
    }

    TensorImpl TensorMaker::randn(const Shape& shape) {
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::normal_distribution<data_t> dis(0, 1);
        TensorImpl tensor(shape);
        for (int i = 0; i < tensor.d_size(); ++i)
            tensor.item(i) = dis(gen);
        return tensor;
    }

    TensorImpl TensorMaker::randn_like(const TensorImpl& tensor) {
        return randn(tensor.size());
    }
}