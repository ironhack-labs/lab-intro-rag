![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | RAG from Scratch


## Introduction

A bank's data protection team answers the same privacy questions every day, and the answers are all in one 88-page document: the General Data Protection Regulation (GDPR). In this lab you build a first version of a privacy assistant with retrieval-augmented generation, in plain Python. You load and chunk the GDPR with page and article metadata, embed the chunks with MiniLM, store them in a Chroma collection, retrieve the closest chunks with a metadata filter, and write a grounded prompt that cites its sources and refuses when the GDPR has no answer. Finally, you evaluate the assistant on 10 questions with known answers and decide whether each failure comes from retrieval or from generation. In the extra task, you tune the chunk size or index a document of your own.

The lab follows the **RAG from Scratch** lesson step by step.


## Getting Started

Open the `main.ipynb` file in the `your-code` directory. Follow the instructions, add your code, and answer the questions in the markdown cells. Each challenge ends with a checkpoint that tells you what you should see. The document is in the `data` folder: `gdpr.pdf`.

The lab needs `pip install pypdf sentence-transformers chromadb openai python-dotenv` and the OpenAI API key that Ironhack gives you. Create a file called `.env` in the `your-code` folder with one line, `OPENAI_API_KEY=sk-...`. The repository's `.gitignore` already excludes `.env`: **never write your key in the notebook, and run `git status` before every commit to check that `.env` is not listed.** The embedding model (about 90 MB) downloads the first time you run it, and the whole lab costs about one US cent.

## Deliverables

- `main.ipynb` with your code, its outputs, and your answers.

## Submission

Upon completion, add your deliverables to git. Then commit git and push your branch to the remote.

## Resources

- [pypdf documentation](https://pypdf.readthedocs.io/en/stable/)
- [Sentence Transformers documentation](https://sbert.net/)
- [Chroma documentation: querying and metadata filters](https://docs.trychroma.com/docs/querying-collections/metadata-filtering)
- [OpenAI API reference: Chat Completions](https://platform.openai.com/docs/api-reference/chat)

## Data

`data/gdpr.pdf` is Regulation (EU) 2016/679 of the European Parliament and of the Council (General Data Protection Regulation), as published in the Official Journal of the European Union, L 119, 4 May 2016, downloaded from [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj). © European Union, https://eur-lex.europa.eu. It is reused under [Commission Decision 2011/833/EU](https://eur-lex.europa.eu/eli/dec/2011/833/oj) on the reuse of Commission documents. Only the Official Journal version is authentic, and nothing in this lab is legal advice.
