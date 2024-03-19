# NYT Connections Helper

Author: Amirali Danai (amiralid@umich.edu)

A basic command line LLM helper for the NYT Connections Game. The helper is quite silly and rather useless for the game due to its difficulty, but is a good example of implementing Conversational ReAct Agents with memory and tools.

To use the helper, ensure you have all dependencies, and run:

```
python3 main.py
```

A sample input file is provided in `input.txt`. Edit the file as necessary.

Note: It may take a while for the program to start, as it downloads an embedding model.
Other note: If you modify the list in `input.txt` and include uncommon words, there is a non-zero probability that the program breaks due to the words you include not being in the embedding model. If this happens, consider editing `main.py` to use other embedding models.