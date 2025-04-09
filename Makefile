CXX = g++
# id19path: /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/Eigen
# elactix nova path: /usr/include/eigen-3.4.0
CXXFLAGS = -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -g -std=c++17

OBJ = sum_predictor.o convolutional.o dense.o losses.o activations.o pooling.o network.o reshape.o
OBJ2 = mnist_final.o dataloader.o convolutional.o dense.o losses.o activations.o pooling.o network.o reshape.o
MED_SOURCES = network.cpp \
       dense.cpp \
       convolutional.cpp \
       reshape.cpp \
       activations.cpp \
       pooling.cpp \
       losses.cpp \
       dataloader.cpp \
       medical_classifier.cpp \
       stb_impl.cpp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

MED_OBJS = $(MED_SOURCES:.cpp=.o)
MED_TARGET = medical_classifier

med: $(MED_OBJS)
	$(CXX) $(CXXFLAGS) -o med $(MED_OBJS)

mnist: $(OBJ2)
	$(CXX) $(CXXFLAGS) -o mnist $(OBJ2)

mnist_final.o: mnist_final.cpp
	$(CXX) $(CXXFLAGS) -c mnist_final.cpp

dataloader.o: dataloader.cpp
	$(CXX) $(CXXFLAGS) -c dataloader.cpp

sum_predictor: $(OBJ)
	$(CXX) $(CXXFLAGS) -o sum_predictor $(OBJ)

sum_predictor.o: sum_predictor.cpp network.hpp
	$(CXX) $(CXXFLAGS) -c sum_predictor.cpp

convolutional.o: convolutional.cpp convolutional.hpp
	$(CXX) $(CXXFLAGS) -c convolutional.cpp

dense.o: dense.cpp dense.hpp
	$(CXX) $(CXXFLAGS) -c dense.cpp

losses.o: losses.cpp losses.hpp
	$(CXX) $(CXXFLAGS) -c losses.cpp

activations.o: activations.cpp activations.hpp
	$(CXX) $(CXXFLAGS) -c activations.cpp

pooling.o: pooling.cpp pooling.hpp
	$(CXX) $(CXXFLAGS) -c pooling.cpp

network.o: network.cpp network.hpp
	$(CXX) $(CXXFLAGS) -c network.cpp

reshape.o: reshape.cpp reshape.hpp
	$(CXX) $(CXXFLAGS) -c reshape.cpp

image_loader.o: image_loader.cpp
	$(CXX) $(CXXFLAGS) -c image_loader.cpp

clean:
	rm -f *.o sum_predictor test_img mnist

test_img: image_loader.o
	$(CXX) $(CXXFLAGS) -c test_img_loader.cpp
	$(CXX) $(CXXFLAGS) -o test_img test_img_loader.o image_loader.o
	./test_img

test_loader: test_dataloader.cpp dataloader.cpp
	$(CXX) $(CXXFLAGS) test_dataloader.cpp dataloader.cpp -o test_loader
	./test_loader

# Default rule: if you run `make <something>`, it tries to build `<something>.cpp`
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Run the code you compiled
run-%: %
	./$<

# Something to make testing much easier
test:
	@echo "Usage: make test FILES='main.cpp foo.cpp bar.cpp'"

test_run:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(FILES) -o test_exec
	./test_exec
	rm test_exec
