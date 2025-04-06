CXX = g++
CFLAGS = -I /opt/homebrew/opt/eigen/include/eigen3 -g
EIGENFLAG = -I /opt/homebrew/opt/eigen/include/eigen3

OBJ = sum_predictor.o convolutional.o dense.o losses.o activations.o pooling.o network.o reshape.o

sum_predictor: $(OBJ)
	$(CXX) $(CFLAGS) -o sum_predictor $(OBJ)

sum_predictor.o: sum_predictor.cpp network.hpp
	$(CXX) $(CFLAGS) -c sum_predictor.cpp

convolutional.o: convolutional.cpp convolutional.hpp
	$(CXX) $(CFLAGS) -c convolutional.cpp

dense.o: dense.cpp dense.hpp
	$(CXX) $(CFLAGS) -c dense.cpp

losses.o: losses.cpp losses.hpp
	$(CXX) $(CFLAGS) -c losses.cpp

activations.o: activations.cpp activations.hpp
	$(CXX) $(CFLAGS) -c activations.cpp

pooling.o: pooling.cpp pooling.hpp
	$(CXX) $(CFLAGS) -c pooling.cpp

network.o: network.cpp network.hpp
	$(CXX) $(CFLAGS) -c network.cpp

reshape.o: reshape.cpp reshape.hpp
	$(CXX) $(CFLAGS) -c reshape.cpp

image_loader.o: image_loader.cpp
	$(CXX) $(CFLAGS) -c image_loader.cpp

clean:
	rm -f *.o sum_predictor 

test_img: image_loader.o
	$(CXX) $(CFLAGS) -c test_img_loader.cpp
	$(CXX) $(CFLAGS) -o test_img test_img_loader.o image_loader.o
	./test_img