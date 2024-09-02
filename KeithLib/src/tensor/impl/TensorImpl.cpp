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
}