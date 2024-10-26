# Atlas Exploration

The aim of this repository is to explore atlas as a vector store and implement the rag in an agentic workflow with the potential use of other tools. Since this repository focuses mainly on proof of concept, all exploration is done in a jupyter notebook for greater clarity and flexibility.

## Dev Container Capabilities
Press ```F1``` and click ```Dev Containers: Rebuild and Reopen in Container```\
Full gpu capabilities is provided by default when using the dev container together with the appropriate linting

## Prerequisities:
1) MongoDB uri with atlas (also need to create indexes for the vector searching)
    ```
    {
        "fields": [
            {
            "numDimensions": 1536,
            "path": "embedding",
            "similarity": "cosine",
            "type": "vector"
            },
            {
            "path": "hasCode",
            "type": "filter"
            }
        ]
    }
    ```

2) Openai key (other LLM's are supported as well, but will need to reconfigure the logic. Refer to langchian documentation)

## Overview
### Atlas Workflow
Exploring atlas as a vector store and how it functions with LLM's.
### Agent Workflow