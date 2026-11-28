#pragma once
#include <cstddef>
#include <cstdint>
struct Singer;
extern "C" Singer *singer_new(uint32_t rate);
extern "C" void singer_free(Singer *singer);
extern "C" void singer_process(Singer *singer, uint32_t mask, const float *input, float *output, size_t frames);

constexpr unsigned minorScale(unsigned root) {
    constexpr unsigned minor = (1<<0)|(1<<2)|(1<<3)|(1<<5)|(1<<7)|(1<<8)|(1<<10);
    return ((minor<<root)|(minor>>(12-root))) & 0xfff;
}
