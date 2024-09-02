#pragma once
#include "impl/TensorImpl.h"

namespace keith {

	class Tensor : public Exp<TensorImpl>
	{
		using Exp<TensorImpl>::impl_ptr;
	public:
		Tensor(const Storage& storage, const Shape& shape, const Array<index_t>& stride);
		Tensor(const Storage& storage, const Shape& shape);
		explicit Tensor(const Shape& shape);
		Tensor(const data_t* data, const Shape& shape);
		Tensor(Storage&& storage, Shape&& shape, Array<index_t>&& stride);
		Tensor(const Tensor& other) = default;
		Tensor(Tensor&& other) = default;
		Tensor& operator=(const Tensor& other)
		{
			if (this != &other) {
				impl_ptr = Alloc::unique_construct<TensorImpl>(*other.impl_ptr);
			}
			return *this;
		}
		Tensor& operator=(Tensor&& other) = default;
		~Tensor() = default;
		explicit Tensor(Alloc::NonTrivalUniquePtr<TensorImpl>&& ptr);
	};

}

