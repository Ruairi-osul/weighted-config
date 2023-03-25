# Weighted Configuration Model Generator
A Python package to generate fully connected weighted graphs with the same strength distribution as the input graph. This package extends the functionality of the popular NetworkX Python library, specifically focusing on the generation of random graphs using various optimization methods. The package is designed with performance in mind and follows academic papers to ensure accuracy and reliability.

## Features
- Generate fully connected weighted graphs with preserved strength distribution.
- Supports multiple optimization methods:
    - L-BFGS-B (SciPy)

- Extends the NetworkX library for seamless integration.
- Object-oriented code structure for easy customization.
- Fully tested using the pytest framework.
- Comprehensive documentation.

## Installation
To install the package, simply clone the repository and install the required dependencies:

```bash
git clone https://github.com/Ruairi-osul/weighted-config.git
cd weighted-config
pip install -r requirements.txt
```

## Usage

Here's a simple example to generate a weighted configuration model using the package:

```python
import networkx as nx
from weighted_config.config_models import ScipyOptimizeConfigurationModelGenerator
from weighted_config.utils import GraphGenerator

# Create a fully connected weighted networkx graph (input_graph)
input_graph = GraphGenerator().good_graph(n_nodes=10)

# Generate a graph using SciPy optimization
config_model_generator = ScipyOptimizeConfigurationModelGenerator(input_graph, method='L-BFGS-B')
generated_graph_scipy = config_model_generator.generate()

```

## Tests
The package is fully tested using the pytest framework. To run the tests, execute the following command:

## Contributing
We welcome contributions to the Weighted Configuration Model Generator package! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch with a descriptive name.
3. Make your changes and commit them with a meaningful message.
4. Create a pull request describing your changes and referencing any relevant issues.
5. Ensure that your code passes all the tests.

## License
This project is licensed under the MIT License. See the `LICENSE.txt` file for details.

## Acknowledgements
This package was inspired by the NetworkX library.