# caffe2_test
caffe2 学习

ubuntu 16.04 
CPU/CUDA


## 2018-3-14
    新增：caffe2_split_net
    支持CPU和CUDA；参考caffe2_cpp_tutorial项目写的，功能还出一试验阶段，让大家见笑了。
    分别介绍一下功能： 
        在移动平台使用中，如果图片识别种类增加，就至少要重新训练最后一层全连接成，模型就必须重新下载
    从而导致大量的流量。如果将不变的层分离出来，则不需要做这么多重复下载工作，甚至可以把最后变化的层级
    放到server端以增加灵活性。故此，有这探索。   

	void split_net(const std::string &layer, const std::string &folder, bool force_cpu = false);
    拆分网络为两个部分。
    
	vector<std::string> predict2protobuf(const std::string img_path, vector<std::string> &protoStr);
    用拆分后的第一个网络预测，将输入图片转为中间特征张量，压缩为protobuf格式。

    std::vector<std::vector<std::pair<int, float>>> predict(const vector<std::string> &vect_protobuf);
    用拆分后的第二个网络预测，将第一个网络得到的特征张量，预测为最终结果。