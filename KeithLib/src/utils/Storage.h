#pragma once

#include "Allocator.h"

#include <memory>
#include <vector>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <assert.h>

namespace keith {

	template<typename DType>
	class Array {
	public:
        Array(index_t size) :
            size_(size), d_ptr(Alloc::unique_allocate<DType>(size_ * sizeof(DType))) {}
        Array(std::initializer_list<DType> d_list) : Array(d_list.size()) {
            auto ptr = d_ptr.get();
            for (auto d : d_list) {
                *ptr = d;
                ++ptr;
            }
        }
        Array(std::vector<DType> d_list) : Array(d_list.size()) {
            auto ptr = d_ptr.get();
            for (auto d : d_list) {
                *ptr = d;
                ++ptr;
            }
        }
        Array(const Array<DType>& other) :
            size_(other.size()), d_ptr(Alloc::unique_allocate<DType>(size_ * sizeof(DType))) {
            std::memcpy(this->d_ptr.get(), other.d_ptr.get(), size_ * sizeof(DType));
        }
        Array(const DType* arr, index_t size) :
            size_(size), d_ptr(Alloc::unique_allocate<DType>(size_ * sizeof(DType))) {
            std::memcpy(this->d_ptr.get(), arr, size_ * sizeof(DType));
        }
        Array(Array<DType>&& other)  noexcept = default;

        ~Array() = default;
    public:
        DType& operator[](index_t idx) { return d_ptr.get()[idx]; }
        DType operator[](index_t idx) const {
            if (idx >= size_) {
                std::cout << idx << " " << size_ << std::endl;
            }
            assert(idx < size_);
            return d_ptr.get()[idx];
        }
    public:
        int size() const { return this->size_; }
        void memset(int value) const { std::memset(d_ptr.get(), value, size_ * sizeof(DType)); }
        void fill(DType value) const { std::fill_n(d_ptr.get(), size_, value); }
    private:
        index_t size_;
        Alloc::TrivalUniquePtr<DType> d_ptr;
	};

    typedef double data_t;

	class Storage
	{
    public:
        explicit Storage(index_t size);
        Storage(const Storage& other, index_t offset);
        Storage(index_t size, data_t value);
        Storage(const data_t* data, index_t size);
        Storage(const std::initializer_list<data_t>& list);

        explicit Storage(const Storage& other) = default;
        explicit Storage(Storage&& other) = default;

        ~Storage() = default;

        Storage& operator=(const Storage& other) = delete;

        data_t operator[](index_t idx) const { return f_ptr[idx]; }
        data_t& operator[](index_t idx) { return f_ptr[idx]; }
        [[nodiscard]] index_t offset() const { return f_ptr - b_ptr->data_; }
        index_t size_;
    private:
        struct Data {
            data_t data_[1];
        };
        std::shared_ptr<Data> b_ptr;
        data_t* f_ptr;
	};

}

