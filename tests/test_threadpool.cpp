#include "threadpool.h"

#include <gtest/gtest.h>
#include <random>



TEST(Threadpool, CheckInit) {
    auto& pool = Threadpool::get_instance();
}

TEST(Threadpool, CheckSubmit) {
    auto& pool = Threadpool::get_instance();

    auto future = pool.submit([]() { return 42; });
    auto result = future.get();

    ASSERT_EQ(result, 42);
}

TEST(Threadpool, CheckManyTasks) {
    auto& pool = Threadpool::get_instance();
    std::vector<std::future<int>> futures;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < 100; i++) {

        // Test tasks that may not be executed in the given order
        auto future = pool.submit([i, &gen]() { 
            std::uniform_int_distribution<> dis(0, 100);
            std::this_thread::sleep_for(std::chrono::milliseconds(dis(gen)));
            return i; 
        });

        futures.push_back(std::move(future));
    }

    for (int i = 0; i < 100; i++) {
        auto result = futures[i].get();
        ASSERT_EQ(result, i);
    }

}


