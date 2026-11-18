// armN64's stand-in for SPIRV-Tools' hex_float.h and bitutils.h (Apache-2.0 only), so a build stays
// under the GPL-2.0: just what SpvBuilder.cpp uses, a float's value cast to a smaller float's bits,
// rounding toward zero. armN64's shaders make no 16- or 8-bit float constants.
//
// Copyright (c) 2026 Magnominus
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cmath>
#include <cstdint>

namespace spvutils {

enum round_direction { kRoundToZero, kRoundToNearestEven, kRoundToPositiveInfinity, kRoundToNegativeInfinity };

// A small float's bits: Exponent and Mantissa bits wide, with infinities when Infinite (else its
// largest exponent is a number too).
template <int Exponent, int Mantissa, bool Infinite>
class SmallFloat {
public:
	SmallFloat(uint32_t bits = 0) : bits(bits) {}
	uint32_t get_value() const { return bits; }
	uint32_t bits;
};

using Float16 = SmallFloat<5, 10, true>;
using FloatE5M2 = SmallFloat<5, 2, true>;
using FloatE4M3 = SmallFloat<4, 3, false>;
using FloatE3M2 = SmallFloat<3, 2, false>;
using FloatE2M3 = SmallFloat<2, 3, false>;
using FloatE2M1 = SmallFloat<2, 1, false>;

template <typename T>
class FloatProxy {
public:
	FloatProxy(T value) : value(value) {}
	template <typename U>
	FloatProxy(U value) : value(static_cast<T>(value)) {}
	T getAsFloat() const { return value; }
	T value;
};

template <typename T>
class HexFloat {
public:
	HexFloat(T value) : proxy(value) {}
	T value() const { return proxy; }

	// The value as a smaller float, toward zero; past its largest, its largest.
	template <int Exponent, int Mantissa, bool Infinite>
	void castTo(HexFloat<FloatProxy<SmallFloat<Exponent, Mantissa, Infinite>>>& out, round_direction) const {
		float const f = proxy.getAsFloat();
		uint32_t const sign = std::signbit(f) ? 1u << (Exponent + Mantissa) : 0;
		uint32_t const all = (1u << (Exponent + Mantissa)) - 1;
		int const bias = (1 << (Exponent - 1)) - 1;
		int const top = Infinite ? (1 << Exponent) - 2 : (1 << Exponent) - 1;
		uint32_t bits;
		if (std::isnan(f)) {
			bits = all;
		} else if (std::isinf(f)) {
			bits = Infinite ? uint32_t(top + 1) << Mantissa : all;
		} else {
			float const a = std::fabs(f);
			int e;
			std::frexp(a, &e);
			int const biased = a == 0.0f ? 0 : e - 1 + bias;
			if (a == 0.0f) {
				bits = 0;
			} else if (biased > top) {
				// E4M3's all-ones is its NaN: the largest number is one below.
				bool const nanAtTop = !Infinite && Exponent == 4 && Mantissa == 3;
				bits = uint32_t(top) << Mantissa | ((1u << Mantissa) - (nanAtTop ? 2 : 1));
			} else if (biased <= 0) {
				bits = uint32_t(std::ldexp(a, Mantissa - 1 + bias));
			} else {
				bits = uint32_t(biased) << Mantissa | uint32_t(std::ldexp(a, Mantissa - (e - 1)) - std::ldexp(1.0f, Mantissa));
			}
		}
		out.proxy = FloatProxy<SmallFloat<Exponent, Mantissa, Infinite>>(SmallFloat<Exponent, Mantissa, Infinite>(sign | bits));
	}

	T proxy;
};

}  // namespace spvutils
