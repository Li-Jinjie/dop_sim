//
// Created by lijinjie on 23-3-17.
//
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include <torch/script.h>
#include <iostream>

int main(int argc, char **argv) {

//    // ROS based code
//    ros::init(argc, argv, "dop_qd_sim");
//
//    ros::NodeHandle n;
//
//    ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
//
//    int freq = 100;
//
//    ros::Rate loop_rate(freq);

    // Torch based code
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
//        module = torch::jit::load("./models/mul_qd_model.pt");
        module = torch::jit::load("/home/lijinjie/ljj_ws/src/dop_qd_sim/models/mul_qd_model.pt");
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "Loading Torch model is ok\n";

    // Create a vector of inputs.
    int num_agent = 1000;
    std::vector<torch::jit::IValue> inputs;

    torch::Tensor state_init = torch::zeros({num_agent, 31, 1}, dtype(torch::kFloat64)).to(torch::kCUDA);
    state_init.slice(/*dim=*/0, /*start=*/0, /*end=*/num_agent).slice(/*dim=*/1, /*start=*/9, /*end=*/10).slice(/*dim=*/
            2, /*start=*/0, /*end=*/1) = 1.0;  // ew
    inputs.push_back(state_init);

    // please change this python code to C++: rate_cmd = torch.zeros([num_agent, 4, 1], dtype=torch.float64)
    torch::Tensor rate_cmd = torch::zeros({num_agent, 4, 1}, dtype(torch::kFloat64)).to(torch::kCUDA);;
    inputs.push_back(rate_cmd);

    inputs.push_back(double(1 / double(0.02)));

    // loop
    int count = 0;
    clock_t time_pre = clock();
//    while (ros::ok()) {
    while (1) {

//        std_msgs::String msg;
//        std::stringstream ss;
//        ss << "hello world " << count;
//        msg.data = ss.str();
//        ROS_INFO("%s", msg.data.c_str());
//        chatter_pub.publish(msg);

        // ----- torch -----
        // Execute the model and turn its output into a tensor.
        at::Tensor output = module.forward(inputs).toTensor();

//        std::stringstream ss;
//        ss << "round " << count;
//        ROS_INFO("%s", ss.str().c_str());


        // -----------------


//        ros::spinOnce();
//        loop_rate.sleep();
        ++count;

        // time count
        if (count % 500 == 0 && count != 0) {
            clock_t time_now = clock();
            std::cout << "time cost for 500 round: " << (time_now - time_pre) / double(CLOCKS_PER_SEC) << " s"
                      << std::endl;
            std::cout << "time cost for 1 round average: " << (time_now - time_pre) / double(CLOCKS_PER_SEC) / 500
                      << " s" << std::endl;
            time_pre = time_now;
        }
    }


    return 0;
}
