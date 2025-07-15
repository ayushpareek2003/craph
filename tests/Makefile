CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I../inc
SRCDIR = ../src
TESTDIR = .

# Test targets
all: simple_test comprehensive_test

simple_test: simple_test.cpp $(SRCDIR)/sparse_structs.cpp
	$(CXX) $(CXXFLAGS) simple_test.cpp $(SRCDIR)/sparse_structs.cpp -o simple_test

comprehensive_test: test_sparse_formats.cpp $(SRCDIR)/sparse_structs.cpp
	$(CXX) $(CXXFLAGS) test_sparse_formats.cpp $(SRCDIR)/sparse_structs.cpp -o comprehensive_test

# Run tests
run_simple: simple_test
	./simple_test

run_comprehensive: comprehensive_test
	./comprehensive_test

# Clean
clean:
	rm -f simple_test comprehensive_test

# Help
help:
	@echo "Available targets:"
	@echo "  all                 - Build all test executables"
	@echo "  simple_test         - Build simple test"
	@echo "  comprehensive_test  - Build comprehensive test"
	@echo "  run_simple          - Run simple test"
	@echo "  run_comprehensive   - Run comprehensive test"
	@echo "  clean               - Remove test executables"
	@echo "  help                - Show this help"

.PHONY: all run_simple run_comprehensive clean help 