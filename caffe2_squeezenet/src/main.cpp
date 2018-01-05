#include "caffe2/core/flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"
#include <opencv2/opencv.hpp>
#include <ctime>
#include "caffe2/utils/math.h"
#include <fstream>
#include <iostream>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace caffe2
{

std::unique_ptr<Blob> randomTensor(
	const std::vector<TIndex> &dims,
	CPUContext *ctx)
{
	auto blob = make_unique<Blob>();
	auto *t = blob->GetMutable<TensorCPU>();
	t->Resize(dims);
	math::RandUniform<float, CPUContext>(
		t->size(), -1.0, 1.0, t->template mutable_data<float>(), ctx);
	return blob;
}

void run(char *imgPath)
{
	// 定义初始化网络结构与权重值
	caffe2::NetDef init_net, predict_net;
	DeviceOption op;
	op.set_random_seed(1701);

	std::unique_ptr<CPUContext> ctx_;
	ctx_ = caffe2::make_unique<CPUContext>(op);

	// 读入网络结构文件
	//*
	ReadProtoFromFile("squeezenet/exec_net.pb", &init_net);
	ReadProtoFromFile("squeezenet/predict_net.pb", &predict_net);
	//*/
	/*
	ReadProtoFromFile("bvlc_googlenet/init_net.pb", &init_net);
	ReadProtoFromFile("bvlc_googlenet/predict_net.pb", &predict_net);
	//*/

	// Can be large due to constant fills
	VLOG(1) << "Init net: " << ProtoDebugString(init_net);
	LOG(INFO) << "Predict net: " << ProtoDebugString(predict_net);
	auto predictor = caffe2::make_unique<Predictor>(init_net, predict_net);
	LOG(INFO) << "Checking that a null forward-pass works";

	// 用opencv的方式读入文件
	//cv::Mat bgr_img = cv::imread(imgPath, -1);

	struct stat statbuf;
	lstat(imgPath, &statbuf);
	std::vector<char> buffer(statbuf.st_size);
	FILE *pFile = fopen(imgPath, "r");
	if (pFile == nullptr)
	{
		std::cout << "wwwwww\n";
	}
	//std::ifstream in(imgPath, std::ios::in | std::ios::binary);

	std::fread(&buffer[0], 1, buffer.size(), pFile);

	cv::Mat bgr_img = cv::imdecode(cv::Mat(buffer), CV_LOAD_IMAGE_COLOR);

	// 输入图像大小
	const int predHeight = 224; //256;
	const int predWidth = 224;  //256;
	const int crops = 1;		// crops等于1表示batch的数量为1
	const int channels = 3;		// 通道数为3，表示BGR，为1表示灰度图
	const int size = predHeight * predWidth;
	// 初始化网络的输入，因为可能要做batch操作，所以分配一段连续的存储空间
	std::vector<float> inputPlanar(crops * channels * predHeight * predWidth);

	//save compression params
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //PNG格式图片的压缩级别
	compression_params.push_back(20);
	std::cout << "before resizing, bgr_img.cols=" << bgr_img.cols << ", bgr_img.rows=" << bgr_img.rows << std::endl;
	// resize成想要的输入大小
	{
		//将ROI区域图像保存在image中:左上角x、左上角y、矩形长、宽
		//cvSetImageROI(image,cvRect(200,200,600,200));
		//cv::Mat also can do this
		// Mat mask = Mat::Mat(img, R1, Range::all());
		int height = bgr_img.rows;
		int width = bgr_img.cols;
		const double hscale = ((double)height) / predHeight; // 计算缩放比例
		const double wscale = ((double)width) / predWidth;
		const double scale = hscale < wscale ? hscale : wscale;
		const int newH = predHeight * scale;
		const int newW = predWidth * scale;
		cv::Range Rh((height - newH) / 2, (height + newH) / 2);
		cv::Range Rw((width - newW) / 2, (width + newW) / 2);

		std::cout << "Rh:" << Rh.start << "--" << Rh.end << "\n";
		std::cout << "Rw:" << Rw.start << "--" << Rw.end << "\n";
		bgr_img = cv::Mat(bgr_img, Rh, Rw);

		// // cv::namedWindow("bgr_img");
		// // cv::imshow("bgr_img", bgr_img);
		// //cv::waitKey(0);
		std::cout << "after Croping, bgr_img.cols=" << bgr_img.cols << ", bgr_img.rows=" << bgr_img.rows << std::endl;
		//save						  //这里设置保存的图像质量级别
		cv::imwrite("imgs/ww-crop.jpg", bgr_img, compression_params);
	}
	cv::resize(bgr_img, bgr_img, cv::Size{predWidth, predHeight}, 0, 0, cv::INTER_AREA);

	cv::imwrite("imgs/ww-resize.jpg", bgr_img, compression_params);
	//return;

	std::cout << "after resizing, bgr_img.cols=" << bgr_img.cols << ", bgr_img.rows=" << bgr_img.rows << std::endl;
	// Scale down the input to a reasonable predictor size.
	// 这里是将图像复制到连续的存储空间内，用于网络的输入，因为是BGR三通道，所以有三个赋值
	// 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
	for (auto i = 0; i < predHeight; i++)
	{
		//printf("+\n");
		for (auto j = 0; j < predWidth; j++)
		{
			inputPlanar[i * predWidth + j + 0 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 0];
			inputPlanar[i * predWidth + j + 1 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 1];
			inputPlanar[i * predWidth + j + 2 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 2];
		}
	}
	// 输入是float格式
	//for (auto i = 0; i < predHeight; i++) {
	// 模版的输入格式是float
	//  const float* inData = bgr_img.ptr<float>(i);
	//  for (auto j = 0; j < predWidth; j++) {
	//      inputPlanar[i * predWidth + j + 0 * size] = (float)((inData[j]) * 3 ＋ 0);
	//      inputPlanar[i * predWidth + j + 1 * size] = (float)((inData[j]) * 3 ＋ 1);
	//      inputPlanar[i * predWidth + j + 2 * size] = (float)((inData[j]) * 3 ＋ 2);
	//  }
	//}

	//typedef Tensor<CPUContext> TensorCPU;
	// input就是网络的输入，所以把之前准备好的数据赋值给input就可以了
	caffe2::TensorCPU input;
	input.Resize(std::vector<int>({crops, channels, predHeight, predWidth}));
	input.ShareExternalPointer(inputPlanar.data());

	//Predictor::TensorVector inputVec{inputData->template GetMutable<TensorCPU>()};
	Predictor::TensorVector inputVec{&input};

	Predictor::TensorVector outputVec;
	//predictor->run(inputVec, &outputVec);
	//CAFFE_ENFORCE_GT(outputVec.size(), 0);

	std::clock_t begin = clock(); //begin time of inference
	// 预测
	predictor->run(inputVec, &outputVec);

	//std::cout << "CAFFE2_LOG_THRESHOLD=" << CAFFE2_LOG_THRESHOLD << std::endl;
	//std::cout << "init_net.name()" << init_net.name();

	std::clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

	std::cout << "inference takes " << elapsed_secs << std::endl;

	//读取文本
	std::vector<std::pair<string, string>> imagenet_classes;
	{
		std::fstream fIn("squeezenet/imagenet_words.txt", std::ios::in);
		std::string table, tags;
		while (!fIn.eof())
		{
			fIn >> table;
			std::getline(fIn, tags);
			imagenet_classes.push_back(std::make_pair(table, tags));
		}
		//std::cout << imagenet_classes << std::endl;
	}
	float max_value = 0;
	int best_match_index = -1;
	std::vector<std::pair<string, float>> result;
	// 迭代输出结果，output的大小就是网络输出的大小
	for (auto output : outputVec)
	{
		for (auto i = 0; i < output->size(); ++i)
		{
			result.push_back(std::make_pair(imagenet_classes[i].second, output->template data<float>()[i]));
		}
	}
	// std::sort(result.begin(), result.end(), [](std::pair<string, float> n1, std::pair<string, float> n2) -> int {
	// 	return n1.second > n2.second;
	// });
	std::sort(result.begin(), result.end(), [](auto n1, auto n2) -> int {
		return n1.second > n2.second;
	});
	for (int i = 0, n = result.size() < 10 ? result.size() : 10; i < n; i++)
	{
		printf("%.5f :%s\n", result[i].second, result[i].first.c_str());
	}
	std::cout.unsetf(std::ios::fixed);
	// 这里是用imagenet数据集为例
	std::cout
		<< "predicted result is:" << result[0].first
		<< ", with confidence of " << result[0].second << std::endl;
}
}

int main(int argc, char **argv)
{
	caffe2::GlobalInit(&argc, &argv);
	caffe2::run(argv[1]);
	// This is to allow us to use memory leak checks.
	google::protobuf::ShutdownProtobufLibrary();

	// char a;
	// std::cin >> a;
	return 0;
}