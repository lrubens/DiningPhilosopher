
# the compiler: gcc for C program, define as g++ for C++
CC = g++

# compiler flags:
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
CFLAGS  = -ggdb3 -Wall -fopenmp
NUM_PHILOSOPHERS = 10  # Change accordingly for number of philosophers desired

# the build target executable:
TARGET = main

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp 

run:
	./$(TARGET) $(NUM_PHILOSOPHERS)

clean:
	$(RM) $(TARGET)
