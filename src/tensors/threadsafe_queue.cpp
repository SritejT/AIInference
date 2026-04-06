#include "threadsafe_queue.h"
#include <queue>
#include <mutex>

template<typename T>
void ThreadsafeQueue<T>::push(T new_value) {
    std::lock_guard<std::mutex> guard(lock);
    queue.push(new_value);
}

template<typename T>
bool ThreadsafeQueue<T>::try_pop(T& value) {
    std::lock_guard<std::mutex> guard(lock);
    if (queue.empty()) {
        return false;
    }
    value = queue.front();
    queue.pop();
    return true;
}
