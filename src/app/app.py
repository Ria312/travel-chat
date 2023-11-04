import pandas as pd
import json
import re, os
from bs4 import BeautifulSoup
import unicodedata
import requests
import chainlit as cl
from dotenv import load_dotenv
import torch
torch.device('cpu')
from haystack.document_stores import InMemoryDocumentStore
#import datasets
from haystack.nodes import BM25Retriever, PromptTemplate, AnswerParser, PromptNode
from haystack.pipelines import Pipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor, TextConverter

load_dotenv(".env")
index_path = 'my_faiss_index.faiss'
config_path = 'my_faiss_index.json'
path = 'countrytravelinfo.json'
with open(path, encoding="utf8") as f:
    data = json.loads(f.read())

print("Reading data")
sample_data = data[10]['entry_exit_requirements']
cleaned_soup = BeautifulSoup(sample_data, "html.parser" )
soup_text = cleaned_soup.get_text(separator = ' ', strip = True)
clean_text_again = unicodedata.normalize("NFKD",soup_text)

print("-----\n")
def html_cleaning(json_file):
    """ Remove html tags from text
            input: json data
        - These fileds will be cleaned:
            * travel_transportation
            * health
            * local_laws_and_special_circumstances
            * safety_and_security
            * entry_exit_requirements
            * destination_description
            * travel_embassyAndConsulate
            """
    feature_list = [ 'travel_transportation',
                     'health',
                     'local_laws_and_special_circumstances',
                     'safety_and_security',
                     'entry_exit_requirements',
                     'destination_description',
                     'travel_embassyAndConsulate']
    
    for idx, entry in enumerate(json_file):
        for feat in feature_list:
            raw_text = entry[feat]
            cleaned = BeautifulSoup(raw_text, "html.parser" )
            cleaned_text = cleaned.get_text(separator = ' ', strip = True)
            final_text = unicodedata.normalize("NFKD",cleaned_text)
            json_file[idx][feat] = final_text
            
    return json_file

print("indexing pipeline")
html_cleaned_dataset = html_cleaning(data)
df = pd.DataFrame(html_cleaned_dataset)
#document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
#document_store.delete_documents()

if os.path.exists(index_path):
    document_store = FAISSDocumentStore.load()
else:
    document_store = FAISSDocumentStore()
#document_store = FAISSDocumentStore.load(index_path=index_path, config_path=config_path)
indexing_pipeline4 = Pipeline()
text_converter = TextConverter()
print("-----\n")
indexing_pipeline4.add_node(component=text_converter, name="TextConverter", inputs=["File"])
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)
indexing_pipeline4.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
indexing_pipeline4.add_node(component=retriever, name="EmbeddingRetriever", inputs=["PreProcessor"])
indexing_pipeline4.add_node(component=document_store, name="document_store", inputs=['EmbeddingRetriever'])
doc_dir = './doc'
files_to_index = [doc_dir+ "/" + f for f in os.listdir(doc_dir)]
indexing_pipeline4.run_batch(file_paths=files_to_index)
prompt_template = PromptTemplate(prompt = """"Answer the following query based on the provided context. If the context does
                                                not include an answer, reply with 'The data does not contain information related to the question'.\n
                                                Query: {query}\n
                                                Documents: {join(documents)}
                                                Answer: 
                                            """,
                                            output_parser=AnswerParser())


#document_store.save(index_path="my_faiss_index.faiss")
print("Init promot node")
openai_key = os.getenv("OPENAPI_KEY")
prompt_node = PromptNode(model_name_or_path = "gpt-4",
                            api_key = openai_key,
                            default_prompt_template = prompt_template,
                            max_length = 500,
                            model_kwargs={"stream":True})
query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
print("-----\n")

@cl.on_message
async def main(message: cl.Message):
    # Use the pipeline to get a response
    output = query_pipeline.run(query=message.content)

    # Create a Chainlit message with the response
    response = output['answers'][0].answer
    msg = cl.Message(content=response)

    # Send the message to the user
    await msg.send()
