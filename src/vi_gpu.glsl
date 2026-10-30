#version 310 es
// The VI's filters (n64.c n64Picture) on the GPU: each invocation one pixel of the picture. Same
// integer arithmetic as the CPU path; the CPU uploads the framebuffer rows it reads and their hidden
// bits.
precision highp int;
layout(local_size_x = 16, local_size_y = 16) in;

#ifdef VI_SHARED
// Run in the GPU RDP's context: all of RDRAM from its buffer, the hidden bits after it, and the
// picture in RGBA bytes for a texture.
layout(std430, binding = 0) readonly buffer Frame { uint frame[]; };
#define hidden frame
layout(std430, binding = 4) writeonly buffer Picture { uint picture[]; };
#else
layout(std430, binding = 0) readonly buffer Frame { uint frame[]; };
layout(std430, binding = 1) readonly buffer Hidden { uint hidden[]; };
layout(std430, binding = 2) writeonly buffer Picture { uint picture[]; };
#endif

// 0 origin, 1 stride, 2 32-bit pixels, 3 width, 4 height, 5 the frame buffer's first RDRAM byte,
// 6 the hidden buffer's first index, 7 filters (1 antialiasing, 2 dither, 4 divot, 8 gamma)
uniform int u[8];

const uint RDRAM_SIZE = 0x800000u;

// Bytes in the host's order (little endian words).
uint frameByte(uint a) {
	uint o = a - uint(u[5]);
	return (frame[o >> 2] >> ((o & 3u) * 8u)) & 255u;
}

uint hiddenByte(uint i) {
	uint o = i - uint(u[6]);
	return (hidden[o >> 2] >> ((o & 3u) * 8u)) & 255u;
}

// A framebuffer pixel: 8-bit channels and coverage (from the hidden bits for 16-bit pixels, the top
// alpha bits for 32-bit ones).
ivec4 pixel(int x, int y) {
	uint i = uint(y * u[1] + x);
	if (u[2] != 0) {
		uint a = (uint(u[0]) + i * 4u) & (RDRAM_SIZE - 4u);
		return ivec4(int(frameByte(a)), int(frameByte(a + 1u)), int(frameByte(a + 2u)), int((frameByte(a + 3u) >> 5) & 7u));
	}
	uint a = (uint(u[0]) + i * 2u) & (RDRAM_SIZE - 2u);
	uint px = frameByte(a) << 8 | frameByte(a + 1u);
	ivec3 c = ivec3(int((px >> 11) & 31u), int((px >> 6) & 31u), int((px >> 1) & 31u));
	return ivec4(c << 3 | c >> 2, int((px & 1u) << 2 | hiddenByte(a >> 1)));
}

// The workgroup's framebuffer pixels (columns -3 to 18, rows -1 to 16 of its 16 by 16, packed) and
// fetches (columns -1 to 16): neighbours share them instead of each reading its own.
shared uint near[22 * 18];
shared ivec4 fetched[18 * 16];
ivec2 corner;

ivec4 nearPixel(int x, int y) {
	uint v = near[(y - corner.y + 1) * 22 + x - corner.x + 3];
	return ivec4(int(v & 255u), int((v >> 8) & 255u), int((v >> 16) & 255u), int(v >> 24));
}

// The VI's fetch of a pixel, with its coverage as the divot filter sees it: with antialiasing, a
// partly covered pixel moves toward the colour the fully covered pixels around it agree on, by the
// coverage it lacks; with the dither filter, a fully covered one moves a step toward each of its
// eight neighbours.
ivec4 fetch(int x, int y) {
	ivec4 c = nearPixel(x, y);
	if ((u[7] & 1) == 0 || c.w == 7) {
		if ((u[7] & 2) != 0) {
			ivec3 b = c.rgb >> 3, s = ivec3(0);
			for (int dy = -1; dy <= 1; ++dy) {
				for (int dx = -1; dx <= 1; ++dx) {
					if (dx != 0 || dy != 0) {
						ivec3 o = nearPixel(x + dx, y + dy).rgb >> 3;
						s += ivec3(lessThan(b, o)) - ivec3(greaterThan(b, o));
					}
				}
			}
			c.rgb += s;
		}
		return ivec4(c.rgb, 7);
	}
	// The penumbra of the pixel and its fully covered neighbours: for each channel the largest if
	// the pixel has it, else the second largest (repeats count), and the same for the smallest.
	ivec3 hi1 = c.rgb, hi2 = ivec3(-1), lo1 = c.rgb, lo2 = ivec3(256);
	const ivec2 around[6] = ivec2[6](ivec2(-1, -1), ivec2(1, -1), ivec2(-2, 0), ivec2(2, 0), ivec2(-1, 1), ivec2(1, 1));
	for (int k = 0; k < 6; ++k) {
		ivec4 o = nearPixel(x + around[k].x, y + around[k].y);
		if (o.w == 7) {
			hi2 = max(hi2, min(hi1, o.rgb));
			hi1 = max(hi1, o.rgb);
			lo2 = min(lo2, max(lo1, o.rgb));
			lo1 = min(lo1, o.rgb);
		}
	}
	for (int j = 0; j < 3; ++j) {
		int high = c[j] == hi1[j] ? hi1[j] : hi2[j], low = c[j] == lo1[j] ? lo1[j] : lo2[j];
		int col = low + high - (c[j] << 1);
		c[j] = ((((col * (7 - c.w)) + 4) >> 3) + c[j]) & 255;
	}
	return c;
}

// With gamma on, the VI outputs the square root of each channel (scaled to 8 bits).
int gamma(int v) {
	int s = int(sqrt(float(v * 256)));
	while ((s + 1) * (s + 1) <= v * 256) {
		++s;
	}
	while (s * s > v * 256) {
		--s;
	}
	return min(s, 255);
}

void main() {
	corner = ivec2(gl_WorkGroupID.xy) * 16;
	int lastX = min(corner.x + 15, u[3] - 1), lastY = min(corner.y + 15, u[4] - 1);
	for (int i = int(gl_LocalInvocationIndex); i < 22 * 18; i += 256) {
		int nx = corner.x - 3 + i % 22, ny = corner.y - 1 + i / 22;
		if (nx <= lastX + 3 && ny <= lastY + 1) {
			ivec4 c = pixel(nx, ny);
			near[i] = uint(c.r) | uint(c.g) << 8 | uint(c.b) << 16 | uint(c.w) << 24;
		}
	}
	barrier();
	for (int i = int(gl_LocalInvocationIndex); i < 18 * 16; i += 256) {
		int fx = corner.x - 1 + i % 18, fy = corner.y + i / 18;
		if (fx <= lastX + 1 && fy <= lastY) {
			fetched[i] = fetch(fx, fy);
		}
	}
	barrier();
	int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
	if (x >= u[3] || y >= u[4]) {
		return;
	}
	int at = int(gl_LocalInvocationID.y) * 18 + int(gl_LocalInvocationID.x);
	ivec4 l = fetched[at], c = fetched[at + 1], r = fetched[at + 2];
	ivec3 v = c.rgb;
	// The divot filter: each channel the median of a pixel and its neighbours, unless all three are
	// fully covered.
	if ((u[7] & 4) != 0 && (l.w & c.w & r.w) != 7) {
		v = max(min(l.rgb, v), min(max(l.rgb, v), r.rgb));
	}
	if ((u[7] & 8) != 0) {
		v = ivec3(gamma(v.r), gamma(v.g), gamma(v.b));
	}
#ifdef VI_SHARED
	picture[y * u[3] + x] = uint(v.r) | uint(v.g) << 8 | uint(v.b) << 16 | 0xFF000000u;
#else
	picture[y * u[3] + x] = uint(v.r) << 16 | uint(v.g) << 8 | uint(v.b);
#endif
}
