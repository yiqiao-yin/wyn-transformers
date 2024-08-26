# wyn-transformers 🧠✨

A package that allows developers to train a transformer model from scratch, tailored for sentence-to-sentence tasks, such as question-answer pairs.

<details>
<summary>📺 Click here for YouTube Tutorials</summary>

1. [Introduction to wyn-transformers](https://www.youtube.com)
2. [How to Train Your Transformer Model](https://www.youtube.com)
3. [Deploying Models with HuggingFace Hub](https://www.youtube.com)

*More tutorials coming soon!*

</details>

<details>
<summary>📓 Click here for Jupyter Notebook Examples</summary>

1. [Basic Transformer Training Example](./notebooks/Basic_Transformer_Training.ipynb)
2. [Advanced Techniques with Custom Data](./notebooks/Advanced_Techniques_with_Custom_Data.ipynb)
3. [Inference and Evaluation Methods](./notebooks/Inference_and_Evaluation_Methods.ipynb)
4. [Deploying to HuggingFace Hub](./notebooks/Deploying_to_HuggingFace_Hub.ipynb)

*Check the notebooks folder in the repository for more examples.*

</details>


## Description

`wyn-transformers` is a Python package designed to simplify the process of training transformer models from scratch. It's ideal for tasks involving sentence-to-sentence transformations, like building models for question-answering systems.

## Folder Directory 📁

- **Main level**: Contains the `README.md` file.
- **wyn_transformers folder**: 
  - `transformers.py` — Defines the transformer model and helper functions.
  - `inference.py` — Provides utilities for making inferences from a trained model and converting tokens back to text.
  - `push_to_hub.py` — Pushes the trained TensorFlow model to HuggingFace; requires a HuggingFace token.

## Installation 🛠️

To install the `wyn-transformers` package, run:

```bash
! pip install wyn-transformers
```

## Usage 🚀

### Importing the Package

```python
import wyn_transformers
from wyn_transformers.transformers import *

# Hyperparameters
num_layers = 2
d_model = 64
dff = 128
num_heads = 4
input_vocab_size = 8500
maximum_position_encoding = 10000

# Instantiate the Transformer model
transformer = TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding)

# Compile the model
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generate random sample data
sample_data = np.random.randint(0, input_vocab_size, size=(64, 38))

# Fit the model on the random sample data
transformer.fit(sample_data, sample_data, epochs=5)
```

### Using Custom Question-Answer Pairs 📊

You can use a pandas DataFrame to train the model with custom question-answer pairs. Here's an example to get you started:

```python
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Create a sample pandas DataFrame
data = {
    'question': [
        'What is the capital of France?',
        'How many continents are there?',
        'What is the largest mammal?',
        'Who wrote the play Hamlet?'
    ],
    'answer': [
        'The capital of France is Paris.',
        'There are seven continents.',
        'The blue whale is the largest mammal.',
        'William Shakespeare wrote Hamlet.'
    ]
}

# Or read it from a directory
# data = pd.DataFrame("test.csv")

df = pd.DataFrame(data)
df

# Initialize the Tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="")

# Fit the tokenizer on the questions and answers
tokenizer.fit_on_texts(df['question'].tolist() + df['answer'].tolist())

# Convert texts to sequences
question_sequences = tokenizer.texts_to_sequences(df['question'].tolist())
answer_sequences = tokenizer.texts_to_sequences(df['answer'].tolist())

# Pad sequences to ensure consistent input size for the model
max_length = 10  # Example fixed length; this can be adjusted as needed
question_padded = pad_sequences(question_sequences, maxlen=max_length, padding='post')
answer_padded = pad_sequences(answer_sequences, maxlen=max_length, padding='post')

# Combine questions and answers for training
sample_data = np.concatenate((question_padded, answer_padded), axis=0)

# Display the prepared sample data
print("Sample data (tokenized and padded):\n", sample_data)
```

### Converting Tokens Back to Text ✨

Use the `inference` module to convert tokenized sequences back to readable text:

```python
import tensorflow as tf
from wyn_transformers.inference import *

# Testing the function to convert back to text
print("Original token:")
print(question_padded)
print("\nConverted back to text (questions):")
print(sequences_to_text(question_padded, tokenizer))

print("Original token:")
print(answer_padded)
print("\nConverted back to text (answers):")
print(sequences_to_text(answer_padded, tokenizer))
```

### Training with Custom Data 🔄

With the custom tokenized data ready, train the model as before:

```python
# Hyperparameters
num_layers = 2
d_model = 64
dff = 128
num_heads = 4
input_vocab_size = 8500
maximum_position_encoding = 10000

# Instantiate the Transformer model
transformer = TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding)

# Compile the model
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model on the custom sample data
transformer.fit(sample_data, sample_data, epochs=5)
```

## Author 👨‍💻

Yiqiao Yin  
Personal site: [y-yin.io](https://www.y-yin.io/)  
Email: eagle0504@gmail.com  

Feel free to reach out for any questions or collaborations!