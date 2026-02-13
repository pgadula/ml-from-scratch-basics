CC := cc

RAYLIB_INC := third_party/raylib/include
RAYLIB_LIB := third_party/raylib/lib/libraylib.a

CFLAGS := -O2 -Wall -Wextra -I$(RAYLIB_INC) -MMD -MP
CFLAGS_DEBUG := -g -O0 -Wall -Wextra -I$(RAYLIB_INC) -MMD -MP

LDFLAGS := $(RAYLIB_LIB) \
           -lm -ldl -lpthread \
           -lGL -lX11 -lXrandr -lXi -lXcursor -lXinerama

# Programy które chcesz budować
PROGS := knn perceptron svm
PROGS_DEBUG := knn_debug perceptron_debug svm_debug

.PHONY: all debug clean
all: $(PROGS)

debug: $(PROGS_DEBUG)

# -------- Release builds --------
knn: knn.o
	$(CC) -o $@ $^ $(LDFLAGS)

perceptron: perceptron.o
	$(CC) -o $@ $^ $(LDFLAGS)

svm: svm.o
	$(CC) -o $@ $^ $(LDFLAGS)

# -------- Debug builds --------
knn_debug: CFLAGS := $(CFLAGS_DEBUG)
knn_debug: knn_debug.o
	$(CC) -o $@ $^ $(LDFLAGS)

perceptron_debug: CFLAGS := $(CFLAGS_DEBUG)
perceptron_debug: perceptron_debug.o
	$(CC) -o $@ $^ $(LDFLAGS)

svm_debug: CFLAGS := $(CFLAGS_DEBUG)
svm_debug: svm_debug.o
	$(CC) -o $@ $^ $(LDFLAGS)

# -------- Object rules --------
# Release .o from matching .c
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Debug objects (osobne pliki, żeby nie mieszać z release)
%_debug.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(PROGS) $(PROGS_DEBUG) *.o *.d

# Dependency files (auto)
-include *.d
