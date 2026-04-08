#pragma once
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <future>
#include "threadsafe_queue.h"

class MovableFunction {
private:

    // Virtual base class that allows us to call a function of any type (std::packaged_task<T()>) 
    // that might eventually be moved into impl.
    struct ImplBase {
        virtual void call() = 0;
        virtual ~ImplBase() {}
    };

    std::unique_ptr<ImplBase> impl;

    template<typename F>
    struct Impl : ImplBase {
        F f;
        Impl(F&& f_) : f(std::move(f_)) {}
        void call() override { f(); }
    };

public:

    template<typename F>
    MovableFunction(F&& f) : impl(new Impl<F>(std::move(f))) {}

    // Note we only need to support functions of type void(), because we only care about std::packaged_task<T>
    void operator()() { impl->call(); };

    MovableFunction() = default;

    MovableFunction(MovableFunction&& other) : impl(std::move(other.impl)) {}

    MovableFunction& operator=(MovableFunction&& other) {
        impl = std::move(other.impl);
        other.impl = nullptr;
        return *this;
    }

    MovableFunction(const MovableFunction&) = delete;
    MovableFunction& operator=(const MovableFunction&) = delete;
};


class Threadpool {
private:
    inline static std::shared_ptr<Threadpool> pool = nullptr;

    std::atomic_bool done;
    std::vector<std::thread> threads;
    std::mutex queue_mutex;
    ThreadsafeQueue<MovableFunction> work_queue;

    void worker_thread(); 

    Threadpool(size_t num_threads);            

public:

    static std::shared_ptr<Threadpool> get_instance() {

        if (!pool) {
            pool = std::shared_ptr<Threadpool>(new Threadpool(std::thread::hardware_concurrency()));
        }        

        return std::shared_ptr<Threadpool>(pool);
    }

    // Submits a function that takes no arguments to the threadpool.
    // Returns a future that represents the result of the function.
    template<typename FunctionType>
    std::future<typename std::result_of<FunctionType()>::type> submit(FunctionType f) {

        typedef typename std::result_of<FunctionType()>::type ResultType;

        std::packaged_task<ResultType()> task(f);
        std::future<ResultType> res = task.get_future();
        work_queue.push(std::move(task));
        return res;
    }   

    ~Threadpool(); 

};
