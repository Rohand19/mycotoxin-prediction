# DON Concentration Predictor

A deep learning system for predicting DON (Deoxynivalenol) concentration in corn samples using hyperspectral data.

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── attention.py
│   │   └── don_predictor.py
│   ├── preprocessing/
│   │   └── data_processor.py
│   ├── utils/
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── logger.py
│   └── config/
│       └── config.py
├── tests/
│   ├── models/
│   ├── utils/
│   └── integration/
├── docs/
│   ├── api/
│   ├── models/
│   └── examples/
├── notebooks/
│   ├── exploration/
│   └── training/
├── data/
│   └── corn_hyperspectral.csv
├── logs/
├── models/
└── plots/
```

## Features

- Deep learning model with attention mechanism for DON concentration prediction
- Comprehensive data preprocessing pipeline
- Cross-validation with performance metrics
- Feature importance analysis
- Visualization tools for model evaluation
- Memory-efficient training process
- Extensive logging and error handling
- Configuration management system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/don-concentration-predictor.git
cd don-concentration-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the model and training parameters in `src/config/config.py` or create a custom YAML configuration file.

2. Run the training script:
```bash
python src/train.py
```

3. Monitor the training process through the logs in the `logs` directory.

4. View training results and model performance in the `plots` directory.

## Model Architecture

The model uses a combination of dense layers and multi-head self-attention mechanism to process hyperspectral data:

- Input layer for hyperspectral features
- Multi-head self-attention layer for capturing feature interactions
- Multiple dense layers with batch normalization and dropout
- Output layer for DON concentration prediction

## Configuration

The project uses a flexible configuration system that can be modified through:

1. Default configuration in `src/config/config.py`
2. Custom YAML configuration files
3. Runtime parameter updates

Key configuration sections:
- Data preprocessing parameters
- Model architecture settings
- Training hyperparameters
- File paths and directories

## Development

For development:

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{don_concentration_predictor,
  author = {Rohan D},
  title = {DON Concentration Predictor},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Rohand19/don-concentration-predictor}
}
```

## Contact

For questions and feedback, please contact [rohanb2000@gmail.com](mailto:rohanb2000@gmail.com).