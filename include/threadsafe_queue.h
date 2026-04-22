#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadsafeQueue {
private:
    std::queue<T> queue;
    std::mutex lock;
    std::condition_variable data_cond;
    bool finished = false;

public:
    void push(T new_value) {
        std::lock_guard<std::mutex> guard(lock);
        queue.push(std::move(new_value));
        data_cond.notify_one();
    }

    void cleanup() {
        std::lock_guard<std::mutex> guard(lock);
        finished = true;
        data_cond.notify_all();
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

    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lk(lock);
        data_cond.wait(lk, [this] { return !queue.empty() || finished; });
        if (finished) {
            return;
        }
        value = std::move(queue.front());
        queue.pop();
    }


};


