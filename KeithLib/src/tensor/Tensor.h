#pragma once
#include "impl/TensorImpl.h"

namespace keith {

	class Tensor : public Exp<TensorImpl>
	{
		using Exp<TensorImpl>::impl_ptr;
	};

}

