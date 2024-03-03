# LLM Augmentation

Large Language Models boast an incredible breadth of knowledge in just about every domain. In this project, we will explore methods to augment LLMs with custom data and the ability to call functions, the ultimate goal being to improve their performance for specific domain-related tasks.

This repository contains tutorial notebooks aligned with the weekly lessons of the LLM Augmentation project (W24). The notebooks are designed to provide hands-on experience with the technologies and concepts we'll be exploring. 

To get started, simply clone the repository and use the notebooks in your local development environment. If your local environment proves challenging, utilize cloud notebooks like [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/).

## Project Timeline 

| Week | Date  | Weekly Topic                                     | Objective             |
|------|-------|--------------------------------------------------|-----------------------|
| 1    | 2/11  | Setup, Intro to LLMs, & Embeddings               | -                     |
| 2    | 2/18  | Vector Databases & Retrieval Augmented Generation (RAG) | -              |
| -    | -     | Spring Break                                     | -                     |
| 3    | 3/10  | Function Calling & LangChain                     | Form Groups           |
| 4    | 3/17  | Development Time                                 | -                     |
| 5    | 3/24  | Development Time                                 | Group Checkpoint Due  |
| 6    | 3/31  | Building front-ends with Streamlit        | -                     |
| 7    | 4/7   | Development Time                                 | -                     |
| 8    | 4/14  | Final Expo Prep                                  | Final Deliverable Due |
| -    | -     | Final Project Exposition (4/15)                  | -                     |

## API Keys
API keys are an essential part of the project. Everyone will be provided OpenAI API keys, and other keys may be available upon request if necessary for deliverables.

To use API keys in your development environment, either set them as System Environment Variables, or create a `.env` file in your local folder, and set your API key environment variables there. Below is an example of a `.env` file, and Python code pulling an API key from the file.

Your `.env` file should look this:
```py
# .env
OPENAI_API_KEY=your_api_key
OTHER_API_KEYS=...
...
```

To pull environment variables from the `.env` into your code, you will want to use the `dotenv` Python library, like so.
```py
from dotenv import load_dotenv
import os

load_dotenv()
```

Once the above cell is ran, all environment variables from the `.env` variable are loaded into your Notebook environment's variables. To pull these environment variables where necessary, utilize `os.getenv("OPENAI_API_KEY")`.

When using `openai`, setting the API key should look like this:
```py
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
```

When using `langchain`, `load_dotenv()` should suffice, as LangChain automatically looks for environment variables with the appropriate name. If not, do the following:
```py
X_API_KEY = os.getenv("API_KEY_NAME")
# then, pass the API KEY variable where necessary
```

__**Do not**__ hardcode the API keys into your code or include the `.env` file in a Git commit.
