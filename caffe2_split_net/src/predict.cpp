#include "predict.h"

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

template<typename Context> 
PredictNet<Context>::PredictNet(const std::string init_net_path,const std::string predict_net_path)
//PredictNet<Context>::PredictNet(const char *init_net_path, const char *predict_net_path)
	: workspace(nullptr)
{
// #ifdef WITH_CUDA
// 	DeviceOption option;
// 	option.set_device_type(CUDA);
// 	new CUDAContext(option);
// #endif

    //读取文本
    {
        std::fstream fIn("my_googlelenet/classes.txt", std::ios::in);
        int index;
        std::string tags;
        while (!fIn.eof())
        {
            fIn >> index;
            std::getline(fIn, tags);
            imagenet_classes[index] = tags;
        }
        //std::cout << imagenet_classes << std::endl;
    }

	//载入部署模型
	CAFFE_ENFORCE(ReadProtoFromFile(init_net_path, &init_net_def));
	CAFFE_ENFORCE(ReadProtoFromFile(predict_net_path, &predict_net_def));
	setDevice();    //针对CPU和GPU有不同操作(特化版本)
	
	workspace.RunNetOnce(init_net_def); //网络初始化
	predict_net = CreateNet(predict_net_def,&workspace); //创建判别器
}
template<typename Context> 
PredictNet<Context>::~PredictNet()
{
}

template <> 
void PredictNet<CPUContext>::setDevice()
{
	init_net_def.mutable_device_option()->set_device_type(CPU);
	predict_net_def.mutable_device_option()->set_device_type(CPU);	
}
template <> 
void PredictNet<CUDAContext>::setDevice()
{
	init_net_def.mutable_device_option()->set_device_type(CUDA);
	predict_net_def.mutable_device_option()->set_device_type(CUDA);
}

template<typename Context> 
std::vector< std::vector<std::pair<int, float>> >
PredictNet<Context>::predict(const vector<std::string> &vect_protobuf)
{
    //先准备好数据
    CAFFE_ENFORCE(vect_protobuf.size()!=0);
    std::vector<TensorCPU > tensors(vect_protobuf.size());//尽量在栈区开辟变量空间，避免内存泄漏
    TensorDeserializer<CPUContext> deserializer;
    //先反序列化
    for (int i=0;i< vect_protobuf.size();i++)
    {
        TensorProtos protos;
        CAFFE_ENFORCE(protos.ParseFromString(vect_protobuf[i]));
        CAFFE_ENFORCE(protos.protos_size() == 1);
        if (protos.protos(0).has_device_detail())
            protos.mutable_protos(0)->clear_device_detail();
        deserializer.Deserialize(protos.protos(0), &tensors[i]);
       // deserialize(protos.protos(0), &tensors[i]);
    }
    //合并张量
    auto t_dims=tensors[0].dims();
    CAFFE_ENFORCE(t_dims.size()==3);
    std::vector<TIndex> dims_input({(TIndex)tensors.size(), t_dims[0], t_dims[1], t_dims[2]});
    std::vector<float> data; //张量的实际存储内存
    data.reserve(tensors.size()*tensors[0].size());//预分配空间
    for (auto &t : tensors) {
        data.insert(data.end(), t.template data<float>(), t.template data<float>()+t.size());
    }
    //TensorType tensorX(dims_input, data, NULL);
    TensorType tensorX = TensorType(TensorCPU(dims_input, data, NULL)); //生成CPU格式后，再copy到GPU的显存！
    std::vector<TensorType> inputVec{tensorX};  // inputVec.push_back(&tensors[i]);
   // std::vector<TensorType *> outputVec;
   std::vector<TensorCPU> outputVec;

	{
    	//split_predictor->run(inputVec, &outputVec); // 预测
		 CAFFE_ENFORCE(inputVec.size() <= predict_net_def.external_input_size());
		for (auto i = 0; i < inputVec.size(); ++i) {
			auto* blob = workspace.CreateBlob(predict_net_def.external_input(i));  //在workspace中提取blob
			CAFFE_ENFORCE(blob, "Blob: ", predict_net_def.external_input(i), " does not exist");
			auto* tensor = blob->template GetMutable<TensorType>();
			tensor->ResizeLike(inputVec[i]);
			tensor->ShareData(inputVec[i]);
		}

        std::cout << "=================\n" << std::endl;
        std::clock_t begin = clock();               //begin time of inference
		predict_net->Run();
        std::clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "---------------------\ninference takes " << elapsed_secs << std::endl;

		//outputVec.resize(predict_net_def.external_output_size()); //修正大小
		for (auto i = 0; i < predict_net_def.external_output_size(); ++i) {
			auto* blob = workspace.GetBlob(predict_net_def.external_output(i));
  			CAFFE_ENFORCE(blob, "Blob: ", predict_net_def.external_output(i), " does not exist");
  		 	//outputVec[i] = blob->template GetMutable<TensorType>();
            //outputVec[i]= TensorCPU(blob->template Get<TensorType>());
            outputVec.push_back(TensorCPU(blob->template Get<TensorType>()));  //TensorCPU的拷贝构造函数已经禁用，只能使用转移函数(move)?
		}
	}
    std::cout<<outputVec.size() <<std::endl;
    CAFFE_ENFORCE(outputVec.size()==1);
    //TensorType* output = outputVec[0];
    //TensorCPU* output = outputVec[0];
    std::vector<TIndex> dimsOutput = outputVec[0].dims();
    CAFFE_ENFORCE(dimsOutput.size()==2);
    std::vector< std::vector<std::pair<int, float>> > results(dimsOutput[0]);
    for(auto &result:results)
    {
        for (auto i = 0; i < dimsOutput[1]; ++i)
        {
            result.push_back(std::make_pair(i, outputVec[0].template data<float>()[i]));
        }
    } 
    //for(auto &result:results)
    for(int k=0;k< results.size() && k<2 ;k++) 
    {
        auto result = results[k];
        std::sort(result.begin(), result.end(), [](auto n1, auto n2) -> int { //auto === std::pair<string, float>
            return n1.second > n2.second;
        });
        for (int i = 0, n = result.size() < 10 ? result.size() : 10; i < n; i++)
        {
            std::cout<<std::setiosflags(std::ios::fixed) <<std::setprecision(5)                 \
                << result[i].second<<" : "<<imagenet_classes[result[i].first]<<std::endl;
        }
        std::cout<<"-------------------"<<std::endl;
    }
	return results;
}

template<typename Context> 
//vector<std::string> PredictNet<Context>::predict2protobuf(const vector<std::string> &img_paths,vector<std::string> &protoStr)
vector<std::string> PredictNet<Context>::predict2protobuf(const std::string img_path, vector<std::string> &protoStr)
{
    std::vector<float> vec_data;
    //vec_data.reserve(filenames.size() * 3 * width * height);
    vec_data.reserve(1 * 3 * 224 * 224);

    std::cout<<img_path.c_str()<<"\n";
	// 用opencv的方式读入文件
	cv::Mat bgr_img = cv::imread(img_path, -1);
    /*
	struct stat statbuf;
	lstat(img_path.c_str(), &statbuf);
    CAFFE_ENFORCE(statbuf.st_size!=0);
	std::vector<char> buffer(statbuf.st_size);
	FILE *pFile = fopen(img_path.c_str(), "r");
    CAFFE_ENFORCE(pFile != nullptr);
	std::fread(&buffer[0], 1, buffer.size(), pFile);
    std::fclose(pFile);
	cv::Mat bgr_img = cv::imdecode(cv::Mat(buffer), CV_LOAD_IMAGE_COLOR);
    //*/

	// 输入图像大小
	const int predHeight = 224; //256;
	const int predWidth = 224;  //256;
	const int crops = 1;		// crops等于1表示batch的数量为1
	const int channels = 3;		// 通道数为3，表示BGR，为1表示灰度图
	const int size = predHeight * predWidth;
	// 初始化网络的输入，因为可能要做batch操作，所以分配一段连续的存储空间
	std::vector<float> inputPlanar(crops * channels * predHeight * predWidth);

	//save compression params
	// std::vector<int> compression_params;
	// compression_params.push_back(CV_IMWRITE_JPEG_QUALITY); //PNG格式图片的压缩级别
	// compression_params.push_back(30);
	// resize成想要的输入大小
	{
		//将ROI区域图像保存在image中:左上角x、左上角y、矩形长、宽
		int height = bgr_img.rows;
		int width = bgr_img.cols;
		const double hscale = ((double)height) / predHeight; // 计算缩放比例
		const double wscale = ((double)width) / predWidth;
		const double scale = hscale < wscale ? hscale : wscale;
		const int newH = predHeight * scale;
		const int newW = predWidth * scale;
		cv::Range Rh((height - newH) / 2, (height + newH) / 2);
		cv::Range Rw((width - newW) / 2, (width + newW) / 2);
		//std::cout << "Rh:" << Rh.start << "--" << Rh.end << "\n";
		//std::cout << "Rw:" << Rw.start << "--" << Rw.end << "\n";
		bgr_img = cv::Mat(bgr_img, Rh, Rw);
		// // cv::namedWindow("bgr_img");
		// // cv::imshow("bgr_img", bgr_img);
		// //cv::waitKey(0);
		//cv::imwrite("imgs/ww-crop.jpg", bgr_img, compression_params);
	}
	cv::resize(bgr_img, bgr_img, cv::Size{predWidth, predHeight}, 0, 0, cv::INTER_AREA);
	//cv::imwrite("imgs/ww-resize.jpg", bgr_img, compression_params); //输出图像文件

	// 这里是将图像复制到连续的存储空间内，用于网络的输入，因为是BGR三通道，所以有三个赋值
	// 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
	/*
    for (auto i = 0; i < predHeight; i++)
	{
		for (auto j = 0; j < predWidth; j++)
		{
			//opencv存储结构是RGB作为一个整体存的，caffe的存储结构是R、G、B分别作为一个通道，三者组成一个张量
			inputPlanar[i * predWidth + j + 0 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 0];
			inputPlanar[i * predWidth + j + 1 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 1];
			inputPlanar[i * predWidth + j + 2 * size] = (float)bgr_img.data[(i * predWidth + j) * 3 + 2];
		}
	}//*/
    bgr_img.convertTo(bgr_img, CV_32FC3, 1.0, -128);
    CAFFE_ENFORCE_EQ(bgr_img.channels(), 3);
    CAFFE_ENFORCE_EQ(bgr_img.rows, predHeight);
    CAFFE_ENFORCE_EQ(bgr_img.cols, predWidth);
    // convert NHWC to NCHW
    vector<cv::Mat> buf_channels(3);
    cv::split(bgr_img, buf_channels);
    for (auto &c : buf_channels) {
      vec_data.insert(vec_data.end(), (float *)c.datastart, (float *)c.dataend);
    }


	// TensorType input;
	// input.Resize(std::vector<int>({crops, channels, predHeight, predWidth}));
	// input.ShareExternalPointer(inputPlanar.data());
    std::vector<TIndex> dimsX({crops, channels, predHeight, predWidth});
    TensorType tensorX = TensorType(TensorCPU(dimsX, vec_data, NULL)); //生成CPU格式后，再copy到GPU的显存！
	std::vector<TensorType> inputVec{tensorX};
	std::vector<TensorCPU> outputVec;

    {
    	//split_predictor->run(inputVec, &outputVec); // 预测
		 CAFFE_ENFORCE(inputVec.size() <= predict_net_def.external_input_size());
		for (auto i = 0; i < inputVec.size(); ++i) {
			auto* blob = workspace.GetBlob(predict_net_def.external_input(i));  //在workspace中提取blob
			CAFFE_ENFORCE(blob, "Blob: ", predict_net_def.external_input(i), " does not exist");
			auto* tensor = blob->template GetMutable<TensorType>();
			tensor->ResizeLike(inputVec[i]);
			tensor->ShareData(inputVec[i]);
		}

        std::clock_t begin = clock();               //begin time of inference
		predict_net->Run();
        std::clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "inference takes " << elapsed_secs << std::endl;

        //return get_tensor(predict_net_def.external_output(0));

		//outputVec.resize(predict_net_def.external_output_size()); //修正大小
		for (auto i = 0; i < predict_net_def.external_output_size(); ++i) {
			auto* blob = workspace.GetBlob(predict_net_def.external_output(i));
  			CAFFE_ENFORCE(blob, "Blob: ", predict_net_def.external_output(i), " does not exist");
            outputVec.push_back(TensorCPU(blob->template Get<TensorType>()));
		}
	}


    CAFFE_ENFORCE(outputVec.size()==1);
    TensorCPU &output = outputVec[0];

    //
    //以下，转为protobuf的string格式
    //vector<std::string> retStrs;
    TensorProtos protos;
    TensorProto *data = protos.add_protos();
    data->set_data_type(TensorProto::FLOAT);

    TensorSerializer<CPUContext> serializer;
    std::string value;
    std::vector<TIndex> dims(output.dims().begin() + 1, output.dims().end()); //这里要把自一个维度去掉，因为永远是1
    //std::vector<TIndex> dims(output.dims().begin(), output.dims().end());
    auto single_size = output.dim(0) ? output.size() / output.dim(0) : 0; //这里dim(0)是通道维度
    auto output_data = output.template data<float>();                       //多维数组以vector的形式存储
    for (auto i = 0; i < output.dim(0); i++)
    {
        auto single = TensorCPU(dims, std::vector<float>(output_data, output_data + single_size), NULL);
        output_data += single_size;
        data->Clear();
        serializer.Serialize(single, "", data, 0, kDefaultChunkSize);
        //serialize(data,single);
        protos.SerializeToString(&value);
        protoStr.push_back(std::string(value)); //根据std标准，写入的是value的新备份
        //std::cout << "value:" << i << "--" << value << std::endl;
    }
    //std::cout << "value:" << "------" << value << std::endl;
        std::cout << "++++++++++++++++++" <<  std::endl;
    return protoStr;
}

template<typename Context> 
vector<std::string> PredictNet<Context>::get_tensor(std::string layerName)
{
    auto *pOut = workspace.GetBlob(layerName);
    assert(0 != pOut);
    TensorCPU output = TensorCPU(pOut->Get<TensorCPU>());

    //
    //以下，转为protobuf的string格式
    vector<std::string> retStrs;
    TensorProtos protos;
    TensorProto *data = protos.add_protos();
    data->set_data_type(TensorProto::FLOAT);

    //TensorSerializer<CPUContext> serializer;
    std::string value;
    std::vector<TIndex> dims(output.dims().begin() + 1, output.dims().end()); //这里要把自一个维度去掉，因为永远是1
    //std::vector<TIndex> dims(output.dims().begin(), output.dims().end());
    auto size = output.dim(0) ? output.size() / output.dim(0) : 0; //这里dim(0)是通道维度
    auto output_data = output.data<float>();                       //多维数组以vector的形式存储
    for (auto i = 0; i < output.dim(0); i++)
    {
        auto single = TensorCPU(dims, std::vector<float>(output_data, output_data + size), NULL);
        output_data += size;
        data->Clear();
        //serializer.Serialize(single, "", data, 0, kDefaultChunkSize);
        serialize(data,single);
        protos.SerializeToString(&value);
        retStrs.push_back(value); //根据std标准，写入的是value的新备份
        //std::cout << "value:" << i << "--" << value << std::endl;
    }
    return retStrs;
}
template<typename Context> 
void PredictNet<Context>::split_net(const std::string &layer, const std::string &folder,  bool force_cpu)
{
    NetDef first_init_net, first_predict_net;
    NetDef second_init_net, second_predict_net;
    std::set<std::string> static_inputs = layers_to_set(predict_net_def, layer); //predict.CollectLayers(layer);

    // copy operators
    for (const auto &op : init_net_def.op())
    {
        auto is_first = (static_inputs.find(op.output(0)) != static_inputs.end()); //在layer的下层找到
        auto new_op = (is_first ? first_init_net : second_init_net).add_op();
        new_op->CopyFrom(op);
    }
    for (const auto &op : predict_net_def.op())
    {
        auto is_first = (static_inputs.find(op.output(0)) != static_inputs.end());
        auto new_op = (is_first ? first_predict_net : second_predict_net).add_op();
        new_op->CopyFrom(op);
        if (!force_cpu)
        {
            new_op->set_engine("CUDNN"); // TODO: not here
        }
        else if (new_op->has_engine())
        {
            new_op->clear_engine();
        }
    }

    // copy externals
    if (first_predict_net.op().size())
    {
        // first_predict_net.add_external_input(predict.Input(0));
    }
    if (second_predict_net.op().size())
    {
        // second_predict_net.add_external_input(layer);
    }
    //对predict的初始化，其初始化参数(external_input)为init的输出参数(external_output)
    for (const auto &output : init_net_def.external_output())
    {
        auto is_first = (static_inputs.find(output) != static_inputs.end());
        if (is_first)
        {
            first_init_net.add_external_output(output);
        }
        else
        {
            second_init_net.add_external_output(output);
        }
    }

    //根据从下而上的原则，必须在最前
    if (second_predict_net.op().size())
    {
        //second_predict要加一个名为layer的input
        second_predict_net.add_external_input(layer);
    }
    for (const auto &input : predict_net_def.external_input())
    {
        auto is_first = (static_inputs.find(input) != static_inputs.end());
        if (is_first)
        {
            first_predict_net.add_external_input(input);
        }
        else
        {
            second_predict_net.add_external_input(input);
        }
    }
    if (first_predict_net.op().size())
    {
        //layer必须加入到first的external_output里，并加入second的external_input中，否则无法联系彼此
        first_predict_net.add_external_output(layer);
    }
    if (second_predict_net.op().size())
    {
        //原网络的最终输出值，写到second中
        second_predict_net.add_external_output(predict_net_def.external_output(0));
    }

    //注意：
    //    除此之外，必须在second_init网络中加入"ConstantFill"类型的op，其output为layer
    //    否则，运行时会因为找不到名为layer的blob而报错
    if (second_init_net.op().size())
    {
        /*op {
            output: "pool5/7x7_s1"
            name: ""
            type: "ConstantFill"
            arg {
                name: "shape"
                ints: 1
            }
        }
        */
        auto op = second_init_net.add_op();
        op->set_type("ConstantFill");
        op->add_output(layer);
        // auto arg = op->add_arg();
        // arg->set_name("shape");
        // arg->set_i(1);
        auto arg = op->add_arg();
        arg->set_name("shape");
        arg->add_ints(1);
    }

    if (init_net_def.has_name())
    {
        if (!first_init_net.has_name())
        {
            first_init_net.set_name(init_net_def.name() + "_first");
        }
        if (!second_init_net.has_name())
        {
            second_init_net.set_name(init_net_def.name() + "_second");
        }
    }
    if (predict_net_def.has_name())
    {
        if (!first_predict_net.has_name())
        {
            first_predict_net.set_name(predict_net_def.name() + "_first");
        }
        if (!second_predict_net.has_name())
        {
            second_predict_net.set_name(predict_net_def.name() + "_second");
        }
    }

    WriteProtoToBinaryFile(first_init_net, folder+"/vision_first_init_net.pb");
    WriteProtoToBinaryFile(first_predict_net, folder+"/vision_first_predict_net.pb");

    WriteProtoToBinaryFile(second_init_net, folder+"/vision_second_init_net.pb");
    WriteProtoToBinaryFile(second_predict_net, folder+"/vision_second_predict_net.pb");

    WriteProtoToTextFile(first_init_net, folder+"/vision_first_init_net.pbtxt");
    WriteProtoToTextFile(first_predict_net, folder+"/vision_first_predict_net.pbtxt");

    WriteProtoToTextFile(second_init_net, folder+"/vision_second_init_net.pbtxt");
    WriteProtoToTextFile(second_predict_net, folder+"/vision_second_predict_net.pbtxt");

    //WriteProtoToTextFile(init_net, "my_split/vision_init_net.pbtxt");
}

//返回layer层直线的所有blob节点
template<typename Context> 
std::set<std::string> PredictNet<Context>::layers_to_set(const NetDef &net, const std::string &layer)
{
    std::map<std::string, std::set<std::string>> lookup;
    //遍历op的时候，是从底层开始的，从下往上
    for (auto &op : net.op())
    {
        for (auto &input : op.input())
        {
            for (auto &output : op.output())
            {
                lookup[output].insert(input); //input都是对应的op的output的子节点
            }
        }
    }
    std::set<std::string> result;
    for (std::set<std::string> step({layer}); step.size();)
    {
        //layer层之下的所有节点(blob)全部取出来。。。。。。
        std::set<std::string> next;
        for (auto &l : step)
        {
            if (result.find(l) == result.end())
            { //如果result中没有该节点，则插入；避免同名产生死循环
                result.insert(l);
                for (auto &n : lookup[l])
                { //将该节点的对应op的input，也就是layer直线的所有blob！
                    next.insert(n);
                }
            }
        }
        step = next;
    }
    return result;
}


template<typename Context> 
void PredictNet<Context>::deserialize(const TensorProto& proto, TensorType* tensor)
{
	static TensorDeserializer<Context> deserializer;
     deserializer.Deserialize(proto,tensor);
}
template<typename Context> 
void PredictNet<Context>::serialize(TensorProto* proto, const TensorType& tensor)
{
	static TensorSerializer<CPUContext> serializer;
     serializer.Serialize(tensor, "", proto, 0, kDefaultChunkSize);
}

template<typename Context> 
TensorCPU PredictNet<Context>::preProcess(cv::Mat img)
{
}
template<typename Context> 
vector<float> PredictNet<Context>::postProcess(TensorCPU output)
{
}

//
//	针对template编程，class声明和实现分离，https://www.zhihu.com/question/20630104 和 https://stackoverflow.com/questions/1724036/splitting-templated-c-classes-into-hpp-cpp-files-is-it-possible
//	有详细说明。用以下这个办法貌似兼容性不好，建议是所有实现写到class内部就不会有兼容性问题，不过修改.h后，需要删掉.o文件才会生效！
//	对于GCC，也可以在链接的时候加上这个命令来解决问题：-fno-implicit-templates
template class PredictNet<CPUContext>;
template class PredictNet<CUDAContext>;
