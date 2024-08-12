# Chatbot using PyTorch, NLP, and Neural Networks

🤖 This is a chatbot project built using PyTorch, leveraging natural language processing and neural networks to provide smart and contextual responses to user inputs.

## Features

- **NLP-Powered**: Understands user queries through advanced NLP techniques.
- **Neural Networks**: Utilizes a neural network model for accurate responses.
- **PyTorch Framework**: Built with PyTorch, ensuring efficient and flexible model training.

## How It Works

1. **Data Tokenization**: Tokenize the JSON data to break down sentences into individual words.
2. **Stemming and Lowercasing**: Apply stemming to reduce words to their root forms and convert all text to lowercase for consistency.
3. **Bag-of-Words**: Use the bag-of-words approach to create a numerical representation of text data suitable for model training.

## Packages Used

- `nltk` for natural language processing tasks.
- `numpy` for numerical operations.
- `torch` for building and training neural network models.
- `torch.nn` for constructing neural networks.
- `torch.utils.data` for managing data loading.
- Custom modules `nltk_utils` and `model`.

## Getting Started

### Prerequisites

- Python 3.x

### Installation

1. Clone this repository:

```bash
  git clone https://github.com/codeTun/chatbot.git

```

2. Run the following command to install the packages:

```bash
  pip install -r requirements.txt
```
