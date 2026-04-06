#pragma once
#include <queue>
#include <mutex>

template <typename T>
class ThreadsafeQueue {
private:
    std::queue<int> queue;
    std::mutex lock;

public:
    void push(T new_value);

    bool try_pop(T& value); 
};
