#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/utils/proto_utils.h"
#include <opencv2/opencv.hpp>
#include <ctime>
#include "caffe2/utils/math.h"
#include <fstream>
#include <iostream>


// #include "tools.h"
// #include "LeNet.h" 
#include "predict.h"


int main(int argc, char **argv)
{
	caffe2::GlobalInit(&argc, &argv);

	//caffe2::split_net("my_googlelenet/init_net.pb", "my_googlelenet/predict_net.pb", "pool5/7x7_s1");
	//std::vector<std::string> retVal = caffe2::run_predict2protobuf("tools_split/vision_first_init_net.pb", \
	 		"tools_split/vision_first_predict_net.pb", argv[1]);

	vector<std::string> protoStr;
	PredictNet<caffe2::CPUContext> pred0("tools_split/vision_first_init_net.pbtxt", "tools_split/vision_first_predict_net.pbtxt");
	protoStr = pred0.predict2protobuf(argv[1],protoStr);

	for(int i=0;i<1e4;i++)
	{
		protoStr.push_back(protoStr[0]);  
	} 
	//caffe2::run_predict("tools_split/vision_second_init_net.pbtxt", "tools_split/vision_second_predict_net.pbtxt", retVal);
	PredictNet<caffe2::CPUContext> pred("tools_split/vision_second_init_net.pbtxt", "tools_split/vision_second_predict_net.pbtxt");
	pred.predict(protoStr);

	// std::string layerName("pool5/7x7_s1");
	// PredictNet<caffe2::CPUContext> pred_split("my_googlelenet/init_net.pb", "my_googlelenet/predict_net.pb");
	// pred_split.split_net(layerName,std::string("tools_split2"));

	//caffe2::run(argv[1]);
	// This is to allow us to use memory leak checks. 
	google::protobuf::ShutdownProtobufLibrary();
	return 0;
}