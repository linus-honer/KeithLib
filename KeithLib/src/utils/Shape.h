#pragma once

#include "Allocator.h"
#include "Storage.h"

#include <initializer_list>
#include <ostream>

namespace keith {

	class Shape
	{
    public:
        Shape(std::initializer_list<index_t> dim);
        Shape(const Shape& other, index_t skip);
        Shape(index_t* dim, index_t n_dim);
        Shape(Array<index_t>&& dim);

        Shape(const Shape& dim) = default;
        Shape(Shape&& dim) = default;
        ~Shape() = default;

        [[nodiscard]] index_t d_size() const;
        [[nodiscard]] index_t sub_size(index_t start_dim, index_t end_dim) const;
        [[nodiscard]] index_t sub_size(index_t start_dim) const;
        bool operator==(const Shape& other) const;

        [[nodiscard]] index_t n_dim() const { return _dim.size(); }
        index_t& operator[](index_t idx) { return _dim[idx]; }
        index_t operator[](index_t idx) const { return _dim[idx]; }
        operator const Array<index_t>() const { return this->_dim; }
        friend std::ostream& operator<<(std::ostream& out, const Shape& sh);
    private:
        Array<index_t> _dim;
	};

}

