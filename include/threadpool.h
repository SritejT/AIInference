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
        Impl(F&& f_) : f(f_) {}
        void call() override { f(); }
    };

public:

    template<typename F>
    MovableFunction(F&& f) : impl(new Impl<F>(f)) {};

    // Note we only need to support functions of type void(), because we only care about std::packaged_task<T>
    void operator()() { impl->call(); };

    MovableFunction() = default;

    MovableFunction(MovableFunction&& other) : impl(std::move(other.impl)) {}

    MovableFunction& operator=(MovableFunction&& other) {
        impl = std::move(other.impl);
        return *this;
    }

    MovableFunction(const MovableFunction&) = delete;
    MovableFunction& operator=(const MovableFunction&) = delete;
};


class Threadpool {
private:
    std::atomic_bool done;
    std::vector<std::thread> threads;
    std::mutex queue_mutex;
    ThreadsafeQueue<MovableFunction> work_queue;

    void worker_thread(); 

public:

    Threadpool(size_t num_threads);            

    template<typename FunctionType>
    std::future<typename std::result_of<FunctionType()>::type> submit(FunctionType f); 

    ~Threadpool(); 

};
