CAFFE2_PREFIX=/home/vision/caffe2/caffe2
CXXFLAGS = `pkg-config --cflags opencv eigen3` -Isrc -I${CAFFE2_PREFIX}/include -std=c++14
LIBS=`pkg-config --libs opencv eigen3` -lprotobuf -L/usr/local/lib -lopencv_core -lopencv_highgui -lboost_program_options -L${CAFFE2_PREFIX}/lib -lcaffe2 -lglog
OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

all: squeezenet.bin

squeezenet.bin: $(OBJS)
	$(CXX) $^ $(LIBS) -o ${@}

clean:
	$(RM) *.log 
	$(RM) predictor $(OBJS)
