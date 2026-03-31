#include <mutex>
#include "threadsafe_queue.h"

using namespace std;

template<typename T>
void ThreadsafeQueue<T>::push(T new_value) {
    std::lock_guard<std::mutex> lock(mutex);
    data_queue.push(new_value);
    queue_not_empty.notify_one();
}

template<typename T>
bool ThreadsafeQueue<T>::try_pop(T& value) {
    std::lock_guard<std::mutex> lock(mutex);
    if (data_queue.empty()) {
        return false;
    }

    value = data_queue.front();
    data_queue.pop();
    return true;
}

template<typename T>
bool ThreadsafeQueue<T>::empty() {
    std::lock_guard<std::mutex> lock(mutex);
    return data_queue.empty();
}

