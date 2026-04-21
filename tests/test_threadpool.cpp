#include "threadpool.h"

#include <gtest/gtest.h>
#include <random>

class ThreadpoolTest : public testing::Test {
protected:
    inline static Threadpool& pool = Threadpool::get_instance();
};


TEST_F(ThreadpoolTest, CheckSubmit) {
    auto future = ThreadpoolTest::pool.submit([]() { return 42; });
    auto result = future.get();

    ASSERT_EQ(result, 42);
}

TEST_F(ThreadpoolTest, CheckManyTasks) {
    std::vector<std::future<int>> futures;

     for (int i = 0; i < 100; i++) {

        // Test tasks that may not be executed in the given order
        auto future = ThreadpoolTest::pool.submit([i]() { 
            std::random_device rd;
            std::mt19937 gen(rd());

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


