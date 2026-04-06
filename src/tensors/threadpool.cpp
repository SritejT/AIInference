#include <thread>
#include <atomic>
#include <vector>
#include <future>
#include "threadsafe_queue.h"
#include "threadpool.h"

// Tries to get a task and execute it. If there are no tasks available, yield the thread.
void Threadpool::worker_thread() {
    while (!done) {
        MovableFunction task;
        if (work_queue.try_pop(task)) {
            task();
        } else {
            std::this_thread::yield();
        }
    }
}

// Creates worker threads. If creation fails, we stop all worker threads and rethrow the exception.
Threadpool::Threadpool(size_t num_threads) : done(false) {
    try {

        for (size_t i = 0; i < num_threads; i++) {
            threads.push_back(std::thread(&Threadpool::worker_thread, this));
        }

    } catch (...) {
        done = true;
        throw;
    }
}

// Submits a function that takes no arguments to the threadpool.
// Returns a future that represents the result of the function.
template<typename FunctionType>
std::future<typename std::result_of<FunctionType()>::type> Threadpool::submit(FunctionType f) {

    typedef typename std::result_of<FunctionType()>::type ResultType;

    std::packaged_task<ResultType()> task(f);
    std::future<ResultType> res = task.get_future();
    work_queue.push(std::move(task));
    return res;
}

// Cleanup by ending all worker threads.
Threadpool::~Threadpool() {
    done = true;
    for (auto& t : threads) {
        t.join();
    }
}

