//
// Created by lijinjie on 23-3-17.
//
#include <torch/script.h>
#include <iostream>

int main() {

    // Torch based code
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
//        module = torch::jit::load("./models/mul_qd_model.pt");
        module = torch::jit::load("/home/lijinjie/ljj_ws/src/dop_sim/models/mul_qd_model.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "Loading Torch model is ok\n";

    // Create a vector of inputs.
    int num_agent = 1000;
    float dt_ctl = 0.02;
    float dt_sim = 0.02;

    std::vector<torch::jit::IValue> inputs;

    torch::Tensor state_init = torch::zeros({num_agent, 31, 1}, dtype(torch::kFloat64)).to(torch::kCUDA);
    state_init.slice(/*dim=*/0, /*start=*/0, /*end=*/num_agent).slice(/*dim=*/1, /*start=*/9, /*end=*/10).slice(/*dim=*/
            2, /*start=*/0, /*end=*/1) = 1.0;  // ew


    // please change this python code to C++: rate_cmd = torch.zeros([num_agent, 4, 1], dtype=torch.float64)
    torch::Tensor rate_cmd = torch::zeros({num_agent, 4, 1}, dtype(torch::kFloat64)).to(torch::kCUDA);

    inputs.emplace_back(state_init);
    inputs.emplace_back(rate_cmd);
    inputs.emplace_back(dt_sim);

    // loop
    int count = 0;
    clock_t time_pre = clock();
    int count_round_num = 500;

//    while (ros::ok()) {
    while (count < 5000) {

        // ----- torch -----
        // Execute the model and turn its output into a tensor.
        at::Tensor state = module.forward(inputs).toTensor();

//        // print the shape of Tensor state
//        std::cout << "state shape: " << state.sizes() << std::endl;  // [500, 31, 1]

        inputs.clear();
        inputs.emplace_back(state);
        inputs.emplace_back(rate_cmd);
        inputs.emplace_back(dt_sim);

//        // print the shape of vector inputs
//        std::cout << "inputs size: " << inputs.size() << std::endl;  // 3

        ++count;

        if (count == 500) {
            clock_t time_now = clock();
            std::cout << "Stable running!" << std::endl;
//            std::cout << "time cost for" << count_round_num << "round: "
//                      << (time_now - time_pre) / double(CLOCKS_PER_SEC) << " s"
//                      << std::endl;
//            std::cout << "time cost for 1 round average: "
//                      << (time_now - time_pre) / double(CLOCKS_PER_SEC) / count_round_num
//                      << " s" << std::endl;
            time_pre = time_now;
        }

        if (count == 500 + 2000) {
            clock_t time_now = clock();
//            std::cout << "time cost for" << count_round_num << "round: "
//                      << (time_now - time_pre) / double(CLOCKS_PER_SEC) << " s"
//                      << std::endl;
            std::cout << "time cost for 1 round average: "
                      << (time_now - time_pre) / double(CLOCKS_PER_SEC) / 2000 * 1000
                      << " ms" << std::endl;
            time_pre = time_now;
        }


//        // time count
//        if (count % count_round_num == 0 && count != 0) {
//            clock_t time_now = clock();
//            std::cout << "time cost for" << count_round_num << "round: "
//                      << (time_now - time_pre) / double(CLOCKS_PER_SEC) << " s"
//                      << std::endl;
//            std::cout << "time cost for 1 round average: "
//                      << (time_now - time_pre) / double(CLOCKS_PER_SEC) / count_round_num
//                      << " s" << std::endl;
//            time_pre = time_now;
//        }
    }

    return 0;
}
