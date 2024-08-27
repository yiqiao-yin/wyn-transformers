# wyn-transformers 🧠✨

A package that allows developers to train a transformer model from scratch, tailored for sentence-to-sentence tasks, such as question-answer pairs.

Repo is [here](https://github.com/yiqiao-yin/wyn-transformers).

<details>
<summary>📺 Click here for YouTube Tutorials</summary>

1. [Introduction to wyn-transformers](https://youtu.be/D-bbwlV7arU)
2. [Train on Custom Data Frame](https://youtu.be/IZkJwIXRao4)
3. [How to Fine-tune Transformers](https://youtu.be/RJ-kxr5LMQA)
4. [Push and save model to HuggingFace cloud](https://youtu.be/CtVJuQ1knoY)
5. [Load pre-trained transformer and train again](https://youtu.be/ZgzHg7j73Ms)

*More tutorials coming soon!*

</details>

<details>
<summary>📓 Click here for Jupyter Notebook Examples</summary>

1. [Basic Transformer Training Example](https://github.com/yiqiao-yin/WYNAssociates/blob/main/docs/ref-deeplearning/ex_%20-%20wyn-transformers%20tutorial%20-%20part%201-5.ipynb)

*Check the notebooks folder in the repository for more examples.*

</details>


## Description

`wyn-transformers` is a Python package designed to simplify the process of training transformer models from scratch. It's ideal for tasks involving sentence-to-sentence transformations, like building models for question-answering systems.

## Folder Directory 📁

Here's the folder directory for your `wyn-transformers` package using the specified style:

```
wyn-transformers
├── pyproject.toml
├── README.md
├── wyn_transformers
│   ├── __init__.py
│   ├── transformers.py
│   ├── inference.py
│   └── push_to_hub.py
└── tests
    └── __init__.py
```

- **`pyproject.toml`**: The configuration file for the Poetry package manager, which includes metadata and dependencies for your package.
- **`README.md`**: The markdown file that provides information and instructions about the `wyn-transformers` package.
- **`wyn_transformers`**: The main package directory containing the core Python files.
  - **`__init__.py`**: Initializes the `wyn_transformers` package.
  - **`transformers.py`**: Defines the Transformer model and helper functions.
  - **`inference.py`**: Contains functions for making inferences from the trained model and converting tokens back to text.
  - **`push_to_hub.py`**: Provides functionality to push the trained TensorFlow model to HuggingFace, requiring a HuggingFace token.
- **`tests`**: The directory for test scripts and files.
  - **`__init__.py`**: Initializes the tests package.

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

### Question-Answer

After the model is trained, it is usually desired to make an inference. This way the user can experience the outcome of the model. To do this, we can use the following code to send a question into the function `predict_text`. The function is designed to tokenize the question, make a prediction using the trained model, and convert the numerical output back to texts.

```python
# Test the function with the example input
input_text = "what is the capital of France?"
predicted_response = predict_text(input_text, transformer, tokenizer, max_length=15)
print("Predicted Response:", predicted_response)
```

### Push model to the cloud

When you are at a good stopping point, you can use the following code to push your model to the HuggingFace cloud environment. We provide helper function `push_model_to_huggingface` to assit you to serialize the model to write the artifact, weights, and architecture to the cloud. 

```python
from wyn_transformers.push_to_hub import *

# Example usage:
huggingface_token = "HF_TOKEN_HERE"
account_name = "HF_ACCOUNT_NAME"
model_name = "MODEL_NAME"

# Call the function to push the model
# result = push_model_to_huggingface(huggingface_token, account_name, transformer, model_name)
result = push_model_to_huggingface(huggingface_token, account_name, transformer, model_name, tokenizer)
print(result)
```

### Load Pre-trained Model

When you desire to load the model back, you can use the following code to load your pre-trained transformer model to continue fine-tuning.

```python
from huggingface_hub import hf_hub_download
import tensorflow as tf
import os
import json
import pickle

# Define the Hugging Face model repository path
model_repo_url = f"{account_name}/{model_name}"

# Step 1: Download the model file from Hugging Face
model_filename = f"{model_name}.keras"
model_file_path = hf_hub_download(repo_id=model_repo_url, filename=model_filename, use_auth_token=huggingface_token)

# Step 2: Load the pre-trained model from the downloaded file
pre_trained_transformer = tf.keras.models.load_model(model_file_path, custom_objects={"TransformerModel": TransformerModel})

# Step 3: Compile the model to prepare for further training
pre_trained_transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 4: Reload the tokenizer (if used) by downloading tokenizer files from Hugging Face
tokenizer_config_path = hf_hub_download(repo_id=model_repo_url, filename="tokenizer_config.json", use_auth_token=huggingface_token)
vocab_path = hf_hub_download(repo_id=model_repo_url, filename="vocab.pkl", use_auth_token=huggingface_token)

# Load the tokenizer configuration from the downloaded file
with open(tokenizer_config_path, "r") as f:
    tokenizer_config = json.load(f)

# Recreate the tokenizer using TensorFlow's Tokenizer class
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(
    num_words=tokenizer_config.get("num_words"),
    filters=tokenizer_config.get("filters"),
    lower=tokenizer_config.get("lower"),
    split=tokenizer_config.get("split"),
    char_level=tokenizer_config.get("char_level")
)
tokenizer.word_index = tokenizer_config.get("word_index")
tokenizer.index_word = tokenizer_config.get("index_word")

# Load the vocabulary from the pickle file
with open(vocab_path, "rb") as f:
    tokenizer.word_index = pickle.load(f)

# Clean up downloaded files
os.remove(tokenizer_config_path)
os.remove(vocab_path)
```

## Author 👨‍💻

Yiqiao Yin  
Personal site: [y-yin.io](https://www.y-yin.io/)  
Email: eagle0504@gmail.com  

Feel free to reach out for any questions or collaborations!