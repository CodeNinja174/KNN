HLS ?= /tools/Xilinx/Vitis_HLS/2022.2

CXX = g++
CFLAGS = -Iinclude -Isrc -std=c++14 -O2 -Wall -Wextra
ifneq ("$(wildcard $(HLS)/include)","")
  CFLAGS += -I$(HLS)/include
endif

SRC = $(wildcard src/*.cpp)
TB  = $(wildcard tb/*.cpp)

OBJ = $(SRC:.cpp=.o) $(TB:.cpp=.o)

TARGET = build/sim

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p build
	$(CXX) $(OBJ) $(CFLAGS) -o $(TARGET)

%.o: %.cpp
	$(CXX) -c $< $(CFLAGS) -o $@

run: all
	./$(TARGET)

clean:
	rm -f src/*.o tb/*.o $(TARGET)
