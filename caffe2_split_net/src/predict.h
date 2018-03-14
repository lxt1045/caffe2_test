#ifndef CAFFE2NET_H
#define CAFFE2NET_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>

#include <caffe2/core/context_gpu.h>
#include "caffe2/core/blob_serialization.h"

using namespace caffe2;

template<typename Context>
class PredictNet {
public:
	//typedef Tensor<Context> TensorType;
	using TensorType = Tensor<Context>;

public:
	PredictNet(const std::string init_net_path,const std::string predict_net_path);
	PredictNet(const char *init_net_path, const char *predict_net_path):
			PredictNet(std::string(init_net_path),std::string(predict_net_path)){};
	virtual ~PredictNet();

	std::vector<std::vector<std::pair<int, float>>> predict(const vector<std::string> &vect_protobuf);

	// vector<std::string> predict2protobuf(const vector<std::string> &img_paths, vector<std::string> &protoStr);
	// vector<std::string> predict2protobuf(const std::string img_path, vector<std::string> &protoStr){
	// 	return predict2protobuf({img_path}),protoStr);
	// };
	vector<std::string> predict2protobuf(const std::string img_path, vector<std::string> &protoStr);
	vector<std::string> predict2protobuf(const char *img_path, vector<std::string> &protoStr){
		return predict2protobuf(std::string(img_path),protoStr);
	};
	
	void split_net(const std::string &layer, const std::string &folder, bool force_cpu = false);
	std::vector<std::string> get_tensor(std::string layerName);

protected:
	// virtual TensorCPU preProcess(cv::Mat img) = 0;
	// virtual vector<float> postProcess(TensorCPU output) = 0;
	virtual TensorCPU preProcess(cv::Mat img);
	virtual vector<float> postProcess(TensorCPU output);

	void setDevice();


protected:
	Workspace workspace;
	NetDef init_net_def;
	NetDef predict_net_def;
	unique_ptr<NetBase> predict_net;
	std::map<int,std::string> imagenet_classes;

	void deserialize(const TensorProto& proto, TensorType* tensor);
	void serialize(TensorProto* proto, const TensorType& tensor);
	std::set<std::string> layers_to_set(const caffe2::NetDef &net, const std::string &layer);
};


#endif
