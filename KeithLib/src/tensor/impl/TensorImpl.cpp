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
}