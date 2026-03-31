#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace std;

template<typename T>
class ThreadsafeQueue {
private:
    std::queue<T> data_queue;
    std::mutex mutex;
    std::condition_variable queue_not_empty;

public:

    void push(T new_value); 

    bool try_pop(T& value); 

    bool empty(); 

};
