#pragma once

#include "../../utils/Storage.h"
#include "../../utils/Shape.h"

#include <cmath>
#include <assert.h>

namespace keith {

	namespace op {

        struct Add {
            template<typename LhsType, typename RhsType>
            static data_t eval(Array<index_t>& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                CHECK_EXP_BROADCAST(lhs, rhs);
                return lhs->eval(idx) + rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static Shape size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
            template<typename LhsType, typename RhsType>
            static index_t size(index_t idx, const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                if (idx >= lhs->ndim()) return rhs->size(idx);
                if (idx >= rhs->ndim()) return lhs->size(idx);
                return std::max(lhs->size(idx), rhs->size(idx));
            }
            template<typename LhsType, typename RhsType>
            static index_t n_dim(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return max(lhs->ndim(), rhs->ndim());
            }
        };
        struct Sub {
            template<typename LhsType, typename RhsType>
            static data_t eval(Array<index_t>& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                CHECK_EXP_BROADCAST(lhs, rhs);
                return lhs->eval(idx) - rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static Shape size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct Mul {
            template<typename LhsType, typename RhsType>
            static data_t eval(Array<index_t>& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                CHECK_EXP_BROADCAST(lhs, rhs);
                return lhs->eval(idx) * rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static Shape size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };
        struct Div {
            template<typename LhsType, typename RhsType>
            static data_t eval(Array<index_t>& idx, std::shared_ptr<LhsType> lhs, std::shared_ptr<RhsType> rhs) {
                CHECK_EXP_BROADCAST(lhs, rhs);
                data_t r = rhs->eval(idx);
                CHECK_FLOAT_EQUAL(r, 0, "divisor cannot be zero");
                return lhs->eval(idx) / rhs->eval(idx);
            }
            template<typename LhsType, typename RhsType>
            static const Shape& size(const std::shared_ptr<LhsType>& lhs, const std::shared_ptr<RhsType>& rhs) {
                return lhs->size();
            }
        };

	}

}

