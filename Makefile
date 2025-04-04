CXX = g++
CFLAGS = -I/usr/include/eigen-3.4.0 -g

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

clean:
	rm -f *.o sum_predictor 