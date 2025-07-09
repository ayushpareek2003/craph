# Sparse Matrix Format Tests

This directory contains test files to verify the functionality of the sparse matrix formats (CSR, CSC, COO) and their inheritance structure.

## Test Files

### 1. `simple_test.cpp`
A basic test that demonstrates:
- Creating instances of all three formats (CSR, CSC, COO)
- Using the inherited `Print()` method
- Testing polymorphism with base class pointers

### 2. `test_sparse_formats.cpp`
A comprehensive test suite that includes:
- Testing all three formats with both weighted and unweighted graphs
- Verifying inheritance and polymorphism
- Error handling tests
- Assertions to verify data integrity

## How to Run Tests

### Option 1: Using Makefile (Recommended)
```bash
cd tests
make all                    # Build all test executables
make run_simple            # Run simple test
make run_comprehensive     # Run comprehensive test
make clean                 # Clean up executables
```

### Option 2: Manual Compilation
```bash
# Simple test
g++ -std=c++17 -I../inc simple_test.cpp ../src/sparse_structs.cpp -o simple_test
./simple_test

# Comprehensive test
g++ -std=c++17 -I../inc test_sparse_formats.cpp ../src/sparse_structs.cpp -o comprehensive_test
./comprehensive_test
```

## Expected Output

The tests will show:
- Graph data in different sparse matrix formats
- Verification that inheritance works correctly
- Confirmation that all formats can use the same `Print()` method
- Error handling for invalid files

## Test Coverage

1. **CSR Format**: Tests Compressed Sparse Row format
2. **CSC Format**: Tests Compressed Sparse Column format  
3. **COO Format**: Tests Coordinate format
4. **Inheritance**: Tests polymorphic behavior
5. **Error Handling**: Tests file opening errors

## Graph Data

Tests use the sample graph from `../src/graph.txt`:
```
0 1 4
0 2 6.7
1 3 4
2 3 2
3 4 100
```

This represents a weighted directed graph with 5 nodes and 5 edges. 