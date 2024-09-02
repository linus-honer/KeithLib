#pragma once

#include "../../utils/Shape.h"
#include "../../utils/Storage.h"
#include "../../utils/Allocator.h"
#include "../../utils/Exception.h"
#include "../../utils/Exp.h"

#include <initializer_list>

namespace keith {

	class TensorImpl
	{
	public:
		TensorImpl(const keith::Storage& storage, const keith::Shape& shape, const Array<index_t>& stride);
		TensorImpl(const keith::Storage& storage, const keith::Shape& shape);
		explicit TensorImpl(const keith::Shape& shape);
		TensorImpl(const data_t* data, const keith::Shape& Shape);
		TensorImpl(keith::Storage&& Storage, keith::Shape&& Shape, Array<index_t>&& stride);
		TensorImpl(const TensorImpl& other) = default;
		TensorImpl(TensorImpl&& other) = default;
		template<typename ImplType>
		explicit TensorImpl(const ImplType& impl) : TensorImpl(impl->size()) {
			this->operator=(impl);
		}
    public:
        [[nodiscard]] index_t n_dim() const { return _shape.n_dim(); }
        [[nodiscard]] index_t d_size() const { return  _shape.d_size(); }
        [[nodiscard]] index_t size(index_t idx) const {
            CHECK_IN_RANGE(idx, 0, n_dim(), "Index out of range (expected to be in range of [0, %d), but got %d)",
                n_dim(), idx);
            return _shape[idx];
        }
        [[nodiscard]] const Shape& size() const { return _shape; }
        [[nodiscard]] index_t offset() const { return _storage.offset(); }
        [[nodiscard]] const IndexArray& stride() const { return _stride; }

        bool is_contiguous() const;
    public:
        data_t& operator[](std::initializer_list<index_t> dims);
        data_t operator[](std::initializer_list<index_t> dims) const;
        [[nodiscard]] data_t item() const;
        [[nodiscard]] data_t item(index_t idx) const;
        [[nodiscard]] data_t& item(index_t idx);
        [[nodiscard]] data_t eval(IndexArray idx) const;
        [[nodiscard]] data_t sum() const;
    public:
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> slice(index_t idx, index_t dim = 0) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> slice(index_t start_idx, index_t end_idx, index_t dim) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> transpose(index_t dim1, index_t dim2) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> view(const Shape& Shape) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> permute(std::initializer_list<index_t> dims) const;
        [[nodiscard]] Alloc::NonTrivalUniquePtr<TensorImpl> sum(int idx) const;
    public:
        friend std::ostream& operator<<(std::ostream& out, const TensorImpl& tensor);

        template<typename ImplType>
        TensorImpl& operator=(const ImplType& src) {
            std::vector<index_t> dim_cnt(n_dim(), 0);
            int cnt = 0;
            while (cnt < d_size()) {
                int idx = 0;
                for (int i = 0; i < n_dim(); ++i) {
                    idx += dim_cnt[i] * _stride[i];
                }
                item(idx) = src->eval(dim_cnt);
                for (int i = n_dim() - 1; i >= 0; --i) {
                    if (dim_cnt[i] + 1 < _shape[i]) {
                        dim_cnt[i]++;
                        break;
                    }
                    else {
                        dim_cnt[i] = 0;
                    }
                }
                ++cnt;
            }
            return *this;
        }

    protected:
        Storage _storage;
        Shape _shape;
        IndexArray _stride;
	};

    struct TensorMaker {
        static TensorImpl ones(const Shape& shape);
        static TensorImpl ones_like(const TensorImpl& tensor);
        static TensorImpl zeros(const Shape& shape);
        static TensorImpl zeros_like(const TensorImpl& tensor);
        static TensorImpl rand(const Shape& shape);
        static TensorImpl rand_like(const TensorImpl& tensor);
        static TensorImpl randn(const Shape& shape);
        static TensorImpl randn_like(const TensorImpl& tensor);
    };

}

