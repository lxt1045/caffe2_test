CAFFE2_PREFIX=~/project/caffe2
CUDA_PREFIX=/usr/local/cuda-8.0
CXXFLAGS = `pkg-config --cflags opencv eigen3` -Isrc -I${CAFFE2_PREFIX}/include -I${CUDA_PREFIX}/include -std=c++14 #-DWITH_CUDA

#-fno-implicit-templates :  template 类，函数在class外实现，GCC支持这个命令
LIBS=`pkg-config --libs opencv eigen3` -lprotobuf -L/usr/local/lib -lopencv_core -lopencv_highgui \
 -lboost_program_options -L${CAFFE2_PREFIX}/lib -lcaffe2 -lglog -lcaffe2_gpu -L${CUDA_PREFIX}/lib64 \
 -lcudart -lcurand -lpthread -lm #-fno-implicit-templates
OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

all: split.bin

split.bin: $(OBJS)
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) *.log 
	$(RM) predictor $(OBJS)
