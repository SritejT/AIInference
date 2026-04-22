#include <thread>
#include <atomic>
#include <vector>
#include "threadsafe_queue.h"
#include "threadpool.h"

// Tries to get a task and execute it. If there are no tasks available, yield the thread.
void Threadpool::worker_thread() {
    while (!done) {
        MovableFunction task;
        work_queue.wait_and_pop(task);
        if (done) {
            break;
        }
        task();
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
        work_queue.cleanup();
        throw;
    }
}



// Cleanup by ending all worker threads.
Threadpool::~Threadpool() {
    done = true;
    work_queue.cleanup();
    for (auto& t : threads) {
        t.join();
    }
}

