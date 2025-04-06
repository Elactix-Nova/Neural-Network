CXX = g++
CXXFLAGS = -I /opt/homebrew/opt/eigen/include/eigen3 -g -std=c++17
EIGENFLAG = -I /opt/homebrew/opt/eigen/include/eigen3

OBJ = sum_predictor.o convolutional.o dense.o losses.o activations.o pooling.o network.o reshape.o

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
	rm -f *.o sum_predictor 

test_img: image_loader.o
	$(CXX) $(CXXFLAGS) -c test_img_loader.cpp
	$(CXX) $(CXXFLAGS) -o test_img test_img_loader.o image_loader.o
	./test_img

# Default rule: if you run `make <something>`, it tries to build `<something>.cpp`
%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Run the code you compiled
run-%: %
	./$<