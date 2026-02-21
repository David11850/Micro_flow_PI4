#include <cstdlib>
#include <new>
#ifndef MICROFLOW_ALLOCATOR_HPP
#define MICROFLOW_ALLOCATOR_HPP

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <cstring>

namespace microflow {

/**
 * @brief 内存对齐常量
 *
 * 对于ARM NEON指令，数据需要16字节对齐以实现最佳性能。
 * Cortex-A72的L1缓存行是64字节，所以更高对齐可以提升缓存性能。
 */
constexpr size_t kCacheLineSize = 64;
constexpr size_t kNEONAlignment = 16;
constexpr size_t kDefaultAlignment = 64;  // 使用缓存行对齐作为默认

/**
 * @brief 向上对齐到指定边界
 *
 * @param size 需要对齐的大小
 * @param alignment 对齐边界（必须是2的幂）
 * @return size_t 对齐后的大小
 *
 * @优化点:
 * - 使用位运算替代取模运算，速度更快
 * - 编译期常量可被编译器优化掉
 */
inline constexpr size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief 对齐的内存分配函数
 *
 * @param size 分配大小
 * @param alignment 对齐边界
 * @return void* 对齐的内存指针
 *
 * @优化点:
 * - 使用C++17的aligned_alloc或posix_memalign
 * - 回退到手动对齐的new/delete
 */
inline void* aligned_alloc(size_t size, size_t alignment = kDefaultAlignment) {
    #if defined(__GLIBC__) && (__GLIBC__ >= 2) && (__GLIBC_MINOR__ >= 17)
        // glibc 2.17+ 支持 C++17 的 aligned_alloc
        return std::aligned_alloc(alignment, align_up(size, alignment));
    #elif defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
        // POSIX 兼容系统
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, align_up(size, alignment)) != 0) {
            return nullptr;
        }
        return ptr;
    #else
        // 回退方案：手动对齐
        void* raw_ptr = ::operator new(size + alignment + sizeof(void*));
        void* aligned_ptr = reinterpret_cast<void*>(
            align_up(reinterpret_cast<size_t>(raw_ptr) + sizeof(void*), alignment)
        );
        // 保存原始指针以便释放
        reinterpret_cast<void**>(aligned_ptr)[-1] = raw_ptr;
        return aligned_ptr;
    #endif
}

/**
 * @brief 对齐的内存释放函数
 *
 * @param ptr 要释放的指针
 *
 * @优化点:
 * - 与分配函数配对使用
 * - 处理手动对齐的情况
 */
inline void aligned_free(void* ptr) {
    #if defined(__GLIBC__) && (__GLIBC__ >= 2) && (__GLIBC_MINOR__ >= 17)
        std::free(ptr);
    #elif defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
        std::free(ptr);
    #else
        if (ptr) {
            ::operator delete(reinterpret_cast<void**>(ptr)[-1]);
        }
    #endif
}

/**
 * @brief Bump Pointer分配器 - 核心内存池实现
 *
 * 这是一个极其高效的内存分配策略，特点是：
 * 1. 分配O(1) - 只需要移动指针
 * 2. 无碎片 - 连续分配
 * 3. 批量释放 - 不支持单个释放
 *
 * 适用场景：
 * - 推理过程中间张量的分配（所有张量生命周期一致）
 * - 一次性分配整个前向传播所需内存
 *
 * 树莓派4优化点：
 * - 使用huge_page减少TLB miss（可选）
 * - 内存预取策略
 * - 与CPU缓存行对齐
 */
class BumpPtrAllocator {
public:
    /**
     * @brief 构造函数
     *
     * @param initial_size 初始内存池大小（建议2MB以上）
     * @param slab_size 每次扩容大小（建议与initial_size相同）
     *
     * @设计考量:
     * - MNIST完整推理约需16MB
     * - 小型CNN约需32-64MB
     * - 预分配可避免运行时分配开销
     */
    explicit BumpPtrAllocator(size_t initial_size = 16 * 1024 * 1024,
                              size_t slab_size = 4 * 1024 * 1024);

    /**
     * @brief 析构函数 - 释放所有slab
     */
    ~BumpPtrAllocator();

    // 禁止拷贝和移动
    BumpPtrAllocator(const BumpPtrAllocator&) = delete;
    BumpPtrAllocator& operator=(const BumpPtrAllocator&) = delete;
    BumpPtrAllocator(BumpPtrAllocator&&) = delete;
    BumpPtrAllocator& operator=(BumpPtrAllocator&&) = delete;

    /**
     * @brief 分配对齐的内存
     *
     * @param size 请求大小
     * @param alignment 对齐要求
     * @return void* 分配的内存指针
     *
     * @优化点:
     * - 编译器内联
     * - 仅两条指令（对齐+指针更新）
     * - 零开销分配
     */
    void* allocate(size_t size, size_t alignment = kDefaultAlignment);

    /**
     * @brief 重置分配器到初始状态
     *
     * @优化点:
     * - 不释放内存，只重置指针
     * - 后续分配复用已分配内存
     * - 适用于批处理推理场景
     */
    void reset();

    /**
     * @brief 获取当前已使用内存
     */
    size_t get_used_memory() const { return used_memory_; }

    /**
     * @brief 获取总分配内存
     */
    size_t get_total_memory() const { return total_memory_; }

    /**
     * @brief 获取统计信息
     */
    struct Stats {
        size_t used_memory;
        size_t total_memory;
        size_t num_slabs;
        size_t num_allocations;
    };
    Stats get_stats() const;

private:
    /**
     * @brief 内存块（Slab）结构
     *
     * @设计考量:
     * - 每个slab是一块连续内存
     * - 通过链表管理多个slab
     * - slab之间不需要连续（降低分配失败率）
     */
    struct Slab {
        void* base;          // slab基地址
        void* current;       // 当前分配位置
        size_t size;         // slab总大小
        size_t remaining;    // 剩余大小

        Slab(void* b, size_t s) : base(b), current(b), size(s), remaining(s) {}
    };

    std::vector<Slab> slabs_;      // 所有slab
    size_t slab_size_;             // 每个slab大小
    size_t used_memory_;           // 已使用内存
    size_t total_memory_;          // 总分配内存
    size_t num_allocations_;       // 分配次数统计

    /**
     * @brief 分配新的slab
     *
     * @param min_size 最小需要的空间
     * @return Slab* 新分配的slab
     *
     * @优化点:
     * - 新slab大小至少是请求大小的2倍
     * - 对齐到页边界(4KB)
     */
    Slab* allocate_slab(size_t min_size);
};

/**
 * @brief 全局内存池实例
 *
 * @设计考量:
 * - 单例模式避免多次初始化
 * - 线程局部存储用于多线程场景
 * - 第一次使用时初始化
 */
inline BumpPtrAllocator& get_global_allocator() {
    static thread_local BumpPtrAllocator allocator(
        16 * 1024 * 1024,  // 16MB 初始大小
        4 * 1024 * 1024    // 4MB 扩容大小
    );
    return allocator;
}

/**
 * @brief Arena内存分配器 - 用于Tensor
 *
 * @tparam T 元素类型
 *
 * @设计考量:
 * - 兼容STL容器接口
 * - 使用全局BumpPtrAllocator
 * - 所有分配都是对齐的
 */
template<typename T>
class ArenaAllocator {
public:
    using value_type = T;

    ArenaAllocator() noexcept = default;

    template<typename U>
    ArenaAllocator(const ArenaAllocator<U>&) noexcept {}

    T* allocate(size_t n) {
        // 计算实际分配大小，确保对齐
        size_t size = n * sizeof(T);
        void* ptr = get_global_allocator().allocate(size, alignof(T));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
        // BumpPtrAllocator不支持单独释放
        // 所有内存一起释放，所以这里什么都不做
    }

    template<typename U>
    struct rebind {
        using other = ArenaAllocator<U>;
    };

    template<typename U>
    bool operator==(const ArenaAllocator<U>&) const noexcept { return true; }

    template<typename U>
    bool operator!=(const ArenaAllocator<U>&) const noexcept { return false; }
};

} // namespace microflow

#endif // MICROFLOW_ALLOCATOR_HPP
