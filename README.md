# GoNeurotic

GoNeurotic is a simple neural network implementation in Go. It is designed to learn the 3-input AND gate problem, demonstrating basic neural network training and inference.

## Getting Started

These instructions will help you set up and run the GoNeurotic project on your local machine.

### Prerequisites

Ensure you have the following installed:

- [Go](https://golang.org/doc/install)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/goneurotic.git
    cd goneurotic
    ```

2. **Create the main Go file:**

    Create a file named `main.go` and add the following code:

    ```go
    package main

    import (
        "fmt"
        "math"
        "github.com/yourusername/goneurotic/network" // Adjust the import path based on your actual repository structure
    )

    func main() {
        layers := []int{3, 5, 1}            // Simplified network: 3 input nodes, 5 hidden nodes, 1 output node
        network := network.NewNetwork(layers, 0.1) // Adjusted
