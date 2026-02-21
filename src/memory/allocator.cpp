#include "microflow/allocator.hpp"
#include <stdexcept>
#include <algorithm>

namespace microflow {

BumpPtrAllocator::BumpPtrAllocator(size_t initial_size, size_t slab_size)
    : slab_size_(std::max(initial_size, slab_size))
    , used_memory_(0)
    , total_memory_(0)
    , num_allocations_(0)
{
    // 分配初始slab
    Slab* initial = allocate_slab(initial_size);
    if (!initial) {
        throw std::bad_alloc();
    }
    slabs_.push_back(*initial);
    delete initial;  // 只删除临时对象，保留内存
}

BumpPtrAllocator::~BumpPtrAllocator() {
    // 释放所有slab的内存
    for (const auto& slab : slabs_) {
        aligned_free(slab.base);
    }
}

void* BumpPtrAllocator::allocate(size_t size, size_t alignment) {
    // 对齐请求大小
    size_t aligned_size = align_up(size, alignment);

    // 尝试在当前slab中分配
    for (auto& slab : slabs_) {
        // 计算对齐后的当前指针位置
        size_t current_addr = reinterpret_cast<size_t>(slab.current);
        size_t aligned_addr = align_up(current_addr, alignment);
        size_t padding = aligned_addr - current_addr;

        // 检查是否有足够空间
        if (slab.remaining >= aligned_size + padding) {
            // 更新slab状态
            slab.current = reinterpret_cast<void*>(aligned_addr + aligned_size);
            slab.remaining -= (aligned_size + padding);
            used_memory_ += aligned_size + padding;
            ++num_allocations_;
            return reinterpret_cast<void*>(aligned_addr);
        }
    }

    // 当前slab不够，分配新的
    size_t new_slab_size = std::max(slab_size_, aligned_size * 2);
    Slab* new_slab = allocate_slab(new_slab_size);
    if (!new_slab) {
        throw std::bad_alloc();
    }

    // 同样的分配逻辑
    size_t current_addr = reinterpret_cast<size_t>(new_slab->current);
    size_t aligned_addr = align_up(current_addr, alignment);
    size_t padding = aligned_addr - current_addr;

    new_slab->current = reinterpret_cast<void*>(aligned_addr + aligned_size);
    new_slab->remaining -= (aligned_size + padding);
    used_memory_ += aligned_size + padding;
    ++num_allocations_;

    slabs_.push_back(*new_slab);
    delete new_slab;

    return reinterpret_cast<void*>(aligned_addr);
}

void BumpPtrAllocator::reset() {
    // 重置所有slab到初始状态
    for (auto& slab : slabs_) {
        slab.current = slab.base;
        slab.remaining = slab.size;
    }
    used_memory_ = 0;
    num_allocations_ = 0;
}

auto BumpPtrAllocator::get_stats() const -> Stats {
    return Stats{
        used_memory_,
        total_memory_,
        slabs_.size(),
        num_allocations_
    };
}

BumpPtrAllocator::Slab* BumpPtrAllocator::allocate_slab(size_t min_size) {
    // 对齐到页边界 (4KB)
    size_t aligned_size = align_up(min_size, 4096);

    // 对齐分配
    void* base = aligned_alloc(aligned_size, kDefaultAlignment);
    if (!base) {
        return nullptr;
    }

    total_memory_ += aligned_size;
    return new Slab(base, aligned_size);
}

} // namespace microflow
