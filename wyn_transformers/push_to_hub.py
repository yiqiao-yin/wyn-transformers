import json
import os
import pickle

import tensorflow as tf
from huggingface_hub import HfFolder, create_repo, upload_file
from tensorflow.keras.preprocessing.text import Tokenizer


def push_model_to_huggingface(
    token: str,
    account_name: str,
    transformer_model: tf.keras.Model,
    model_name: str,
    tokenizer: Tokenizer = None,
) -> str:
    """
    Push a TensorFlow Transformer model to Hugging Face Hub, including the model configuration, tokenizer (if available),
    and a model card with YAML metadata.

    Args:
        token (str): Hugging Face token for authentication.
        account_name (str): Hugging Face account username.
        transformer_model (tf.keras.Model): Trained Transformer model to upload.
        model_name (str): Desired model name on Hugging Face Hub.
        tokenizer (Tokenizer, optional): The tokenizer used for training the model. Default is None.

    Returns:
        str: Success or error message based on the result of the upload.
    """
    # Authenticate with Hugging Face Hub
    HfFolder.save_token(token)

    # Create repository name in "account_name/model_name" format
    repo_id = f"{account_name}/{model_name}"

    try:
        # Create the repository on Hugging Face Hub
        create_repo(repo_id, exist_ok=True)

        # Save the model locally with a .keras extension
        model_save_path = "./tmp_model.keras"
        transformer_model.save(model_save_path)

        # Extract model configuration and save to config.json
        config = (
            transformer_model.get_config()
        )  # Get the model configuration as a dictionary
        config_save_path = "./config.json"
        with open(config_save_path, "w") as config_file:
            json.dump(config, config_file)

        # Upload the saved model file to Hugging Face Hub
        upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo=model_name + ".keras",
            repo_id=repo_id,
            repo_type="model",
        )

        # Upload the config.json to Hugging Face Hub
        upload_file(
            path_or_fileobj=config_save_path,
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
        )

        # Optionally, upload the tokenizer if provided
        if tokenizer:
            # Save tokenizer configuration and vocabulary manually
            tokenizer_config_path = "./tokenizer_config.json"
            vocab_path = "./vocab.pkl"

            # Save tokenizer configuration to JSON
            tokenizer_config = {
                "word_index": tokenizer.word_index,
                "index_word": tokenizer.index_word,
                "num_words": tokenizer.num_words,
                "filters": tokenizer.filters,
                "lower": tokenizer.lower,
                "split": tokenizer.split,
                "char_level": tokenizer.char_level,
            }
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f)

            # Save tokenizer word index to a pickle file
            with open(vocab_path, "wb") as f:
                pickle.dump(tokenizer.word_index, f)

            # Upload tokenizer files to Hugging Face Hub
            upload_file(
                path_or_fileobj=tokenizer_config_path,
                path_in_repo="tokenizer_config.json",
                repo_id=repo_id,
                repo_type="model",
            )
            upload_file(
                path_or_fileobj=vocab_path,
                path_in_repo="vocab.pkl",
                repo_id=repo_id,
                repo_type="model",
            )

        # Define YAML metadata for the model card
        yaml_metadata = """
        ---
        language: en
        license: apache-2.0
        tags:
          - question-answering
          - transformer
          - educational
        datasets:
          - custom
        metrics:
          - accuracy
        ---

        """

        # Define a template for the model card with YAML metadata
        model_card_content = """
        # Model Card for {model_name}

        ## Model Details
        This is a Transformer model trained for demonstration purposes. The model was trained using a dataset of question-answer pairs and is designed to understand simple natural language questions.

        ## Intended Use
        This model is intended for educational purposes and demonstrations. It is not suitable for production use or handling sensitive data.

        ## Limitations
        The model may not perform well on out-of-domain questions or complex natural language understanding tasks.

        ## Training Data
        The model was trained on a small dataset of questions and answers created for this demonstration.
        """.format(
            model_name=model_name
        )

        # Combine YAML metadata and model card content
        full_model_card_content = yaml_metadata + model_card_content

        # Save the model card content to a local README.md file
        readme_path = "./README.md"
        with open(readme_path, "w") as f:
            f.write(full_model_card_content)

        # Upload the README.md (model card) to Hugging Face Hub
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

        # Clean up local saved files after uploading
        os.remove(model_save_path)
        os.remove(config_save_path)
        os.remove(readme_path)
        if tokenizer:
            os.remove(tokenizer_config_path)
            os.remove(vocab_path)

        return f"Model, config, tokenizer, and model card pushed successfully to Hugging Face Hub with YAML metadata. \nPlease go to this URL: https://huggingface.co/{repo_id}"

    except Exception as e:
        return f"Error occurred while pushing the model: {str(e)}"
