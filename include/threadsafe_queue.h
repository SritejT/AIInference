#pragma once
#include <queue>
#include <mutex>

template <typename T>
class ThreadsafeQueue {
private:
    std::queue<T> queue;
    std::mutex lock;

public:
    void push(T new_value) {
        std::lock_guard<std::mutex> guard(lock);
        queue.push(std::move(new_value));
    }


    bool try_pop(T& value) {
        std::lock_guard<std::mutex> guard(lock);
        if (queue.empty()) {
            return false;
        }
        value = std::move(queue.front());
        queue.pop();
        return true;
    }
};


