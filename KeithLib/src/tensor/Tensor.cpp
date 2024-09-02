#include "Tensor.h"

#include <memory>

namespace keith {

	Tensor::Tensor(const Storage& storage, const Shape& shape, const Array<index_t>& stride) : Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(storage, shape, stride)) {}
	Tensor::Tensor(const Storage& storage, const Shape& shape) : Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(storage, shape)) {}
	Tensor::Tensor(const Shape& shape) : Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(shape)) {}
	Tensor::Tensor(const data_t* data, const Shape& shape) : Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(data, shape)) {}
	Tensor::Tensor(Storage&& storage, Shape&& shape, Array<index_t>&& stride) : Exp<TensorImpl>(Alloc::unique_construct<TensorImpl>(std::move(storage), std::move(shape), std::move(stride))) {}
	Tensor::Tensor(Alloc::NonTrivalUniquePtr<TensorImpl>&& ptr) : Exp<TensorImpl>(std::move(ptr)) {}

}
