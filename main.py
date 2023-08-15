from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azureml.core import Workspace

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()

try:
    ml_client = MLClient.from_config(credential=credential, path="workspace.json")
except Exception as ex:
    raise Exception(
        "Failed to create MLClient from config file. Please modify and then run the above cell with your AzureML Workspace details."
    ) from ex

ws = Workspace(
    subscription_id=ml_client.subscription_id,
    resource_group=ml_client.resource_group_name,
    workspace_name=ml_client.workspace_name,
)

asset_name = "azureml_docs_mlindex"



from azureml.rag.utils.connections import (
    get_connection_by_name_v2,
    create_connection_v2,
)
aoai_connection_name_gpt = "my_gpt" # connection for gpt-3.5-turbo
aoai_connection_id = None

try:
    aoai_connection_gpt = get_connection_by_name_v2(ws, aoai_connection_name_gpt)
except Exception as ex:
    # Create New Connection
    # Modify the details below to match the `Endpoint` and API key of your AOAI resource, these details can be found in Azure Portal
    raise RuntimeError(
        "Have you entered your AOAI resource details below? If so, delete me!"
    )
    aoai_connection_gpt = create_connection_v2(
        workspace=ws,
        name=aoai_connection,
        category="AzureOpenAI",
        # 'Endpoint' from Azure OpenAI resource overview
        target="https://eastus.api.cognitive.microsoft.com/",
        auth_type="ApiKey",
        credentials={
            # Either `Key` from the `Keys and Endpoint` tab of your Azure OpenAI resource, will be stored in your Workspace associated Azure Key Vault.
            "key": "7f63a73678f34a9f8714a8d1273d7d6f"
        },
        metadata={"ApiType": "azure", "ApiVersion": "2023-05-15"},
    )

aoai_connection_gpt_id = aoai_connection_gpt["id"]



from azureml.rag.mlindex import MLIndex

retriever = MLIndex(
    ml_client.data.get(asset_name, label="latest")
).as_langchain_retriever()



from langchain.chains import RetrievalQA
from azureml.rag.models import init_llm, parse_model_uri

model_config = parse_model_uri(
    "azure_open_ai://deployment/gpt-35-turbo-bani/model/gpt-35-turbo"
)
model_config["api_base"] = aoai_connection_gpt["properties"]["target"]
model_config["key"] = aoai_connection_gpt["properties"]["credentials"]["key"]
model_config["temperature"] = 0.3
model_config["max_retries"] = 3




from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

QUESTION_PROMPT = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

qa = ConversationalRetrievalChain.from_llm(llm=init_llm(model_config),
                                        retriever=retriever,
                                        condense_question_prompt=QUESTION_PROMPT,
                                        memory=memory,
                                        verbose=False)

# while True:
#     query = input('you: ')
#     if query == 'q':
#         break
#     #result = qa({"question": query, "chat_history": chat_history})
#     result = qa({"question": query})
#     print("answer:", result["answer"])

def get_answer(message):
    query = 'you: ' + message + '\n'
    result = qa({"question": query})
    return result["answer"]



