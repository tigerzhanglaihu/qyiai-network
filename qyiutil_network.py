import datetime
import configparser
import panel as pn
from langchain.chains import ConversationalRetrievalChain
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
import os
import time
import random
import string
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pn.extension("perspective")

class networkbot():
    def __init__(self, **params):

        config_file_path = 'network-config.ini'
        # 创建配置解析器对象
        config = configparser.ConfigParser()
        # 读取配置文件
        config.read(config_file_path,encoding="utf8")
        #try:
        #根目录
        self.MODEL_ROOT_PATH = config.get('local', 'MODEL_ROOT_PATH')
        # LLM 下载路径
        self.LLM_MODEL_DOWNLOAD_URL = config.get('local','LLM_MODEL_DOWNLOAD_URL')
        # EMBEDDING 下载路径
        self.EMBEDDING_MODEL_DOWNLOAD_URL = config.get('local','EMBEDDING_MODEL_DOWNLOAD_URL')
        # EMBEDDING 模型路径
        self.EMBEDDING_MODEL_PATH = config.get('local',"EMBEDDING_MODEL_PATH")
        # LLM 本地模型路径
        self.LLM_MODEL_PATH = config.get('local',"LLM_MODEL_PATH")
        # 选用的 Embedding 名称
        self.EMBEDDING_MODEL = config.get('local',"EMBEDDING_MODEL")
        # 本地向量数据库路径
        self.VECTOR_DB_PATH = config.get('local', "VECTOR_DB_PATH")
        # 文件上传路径
        self.FILE_UPLOAD_PATH = config.get('local', "FILE_UPLOAD_PATH")
        # ollma 模型名称 20242424
        self.LLM_OLLAMA_MODEL = config.get('local', "LLM_OLLAMA_MODEL")
        # ollma 模型名称 20242424
        self.LLM_NETWORK_KEY = config.get('local', "LLM_NETWORK_KEY")
        # ollma 模型名称 20242424
        self.LLM_NETWORK_URL = config.get('local', "LLM_NETWORK_URL")

        #except:
        #    print('读取配置文件错误，请检查项目是否完整！')

def get_csv_txt(file_path):
    # Load documents from CSV file
    loader = CSVLoader(file_path=file_path, encoding='utf8')
    docs = loader.load()
    return docs

def get_txtfile_txt(file_path):
    # Load documents from CSV file
    loader = TextLoader(file_path=file_path, encoding='utf8')
    docs = loader.load()
    return docs

def get_pdf_txt(file_path):
    # Load documents from CSV file
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs

def get_filename_without_extension(file_path):
    filename = os.path.basename(file_path)  # 获取文件名
    filename_without_extension = os.path.splitext(filename)[0]  # 去掉后缀
    return filename_without_extension


def faiss_db(directory):
    # 检查指定目录是否存在
    if not os.path.isdir(directory):
        print(f"The specified directory {directory} does not exist.")
        return False
    # 定义要检查的文件名
    file_names = ['index.faiss', 'index.pkl']
    # 检查所有文件是否都存在
    all_files_exist = all(os.path.isfile(os.path.join(directory, file_name)) for file_name in file_names)

    return all_files_exist


# Function to import files into vectorsdb
def import_file_to_vectorsdb(file_path,EMBEDDING_MODEL_PATH,EMBEDDING_MODEL,VECTOR_DB_PATH,vector_db_name):

    try:
        model_name = EMBEDDING_MODEL_PATH + '/' + EMBEDDING_MODEL

        # Log info function
        def log_info(message):
            logging.info(message)

        file_type = os.path.splitext(file_path)[1].lstrip('.')

        if file_type == 'csv':
            docs = get_csv_txt(file_path)
        elif file_type == 'pdf':
            docs = get_pdf_txt(file_path)
        elif file_type == 'txt':
            docs = get_txtfile_txt(file_path)
        else:
            return "E", "Unsupported file type"

        # Split documents into smaller parts
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs)
        log_info(f"Split documents into {len(documents)} parts.")
        # Initialize embeddings model
        embedding = HuggingFaceEmbeddings(model_name=model_name)

        log_info(f"Initialized Hugging Face Embeddings with model: {model_name}")

        timestamp = time.strftime("%Y%m%d%H%M%S")
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        unique_filename =  f"{timestamp}_{random_str}"

        if vector_db_name == '':
            vector_store_path = VECTOR_DB_PATH + '/' + unique_filename
        else:
            vector_store_path = VECTOR_DB_PATH + '/' + vector_db_name
        """
        为文档生成嵌入向量，如果向量数据库已存在则追加，否则创建新的向量数据库。
        """
        if not faiss_db(vector_store_path):
            vector_store = FAISS.from_documents(documents, embedding)
            log_info("Created new vector store with embeddings.")
        else:
            vector_store = FAISS.load_local(vector_store_path, embedding, allow_dangerous_deserialization=True)
            vector_store.add_documents(documents)
            log_info("Added embeddings to existing vector store.")

        vector_store.save_local(vector_store_path)
        log_info(f"Saved vector store to {vector_store_path}")
        return "S", vector_store_path

    except Exception as e:
        error_message = str(e)
        logging.error(error_message)
        return "E", error_message


# 获取向量数据库列表
def get_vectordbs(VECTOR_DB_PATH):
    target_folder = VECTOR_DB_PATH
    #检查当前文件路径是否存在不存在则创建
    # 检查路径是否存在
    if not os.path.exists(target_folder):
        # 如果路径不存在，则创建它
        os.makedirs(target_folder)
        print(f"The directory {target_folder} has been created.")
    vectordbs = [item for item in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, item))]
    if len(vectordbs) == 0:
        vectordbs = []
    logging.info(f"vectordbs: {str(vectordbs)} faiss_path: {str(VECTOR_DB_PATH)}")
    return vectordbs


# 获取检索器
def get_retriever(embeddingsmodel_name, faiss_path):
    logging.info(f"embeddingsmodel_name: {str(embeddingsmodel_name)} faiss_path: {str(faiss_path)}")
    try:
        embedding = HuggingFaceEmbeddings(model_name=embeddingsmodel_name)
        vectordb = FAISS.load_local(faiss_path, embedding, allow_dangerous_deserialization=True)
        return vectordb.as_retriever()
    except Exception as e:
        logging.error(f"An error occurred while loading the retriever: {str(e)}")
        return None


# 保存文件并生成向量数据库
def _save_file(contents, instance: pn.chat.ChatInterface,FILE_UPLOAD_PATH,VECTOR_DB_PATH,EMBEDDING_MODEL_PATH,EMBEDDING_MODEL,vector_db):
    file_input = instance.widgets[1]
    if file_input.value is None:
        return "E: No file selected."

    file_name = file_input.filename
    file_extension = file_name.split(".")[-1]
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_file_name = f"{file_name.split('.')[0]}_{current_time}.{file_extension}" if len(
        file_name) <= 10 else f"UserFile_{current_time}.{file_extension}"

    directory = FILE_UPLOAD_PATH

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, new_file_name)

    if file_extension == 'csv':
        file_contents = contents.to_csv(index=False).encode('utf-8')
        with open(file_path, 'wb') as f:
            f.write(file_contents)
    elif file_extension == 'pdf':
        file_contents = contents.getvalue()
        with open(file_path, 'wb') as f:
            f.write(file_contents)
    elif file_extension == 'txt':
        file_contents = contents
        with open(file_path, 'w') as f:
            f.write(file_contents)

    logging.info(f"文件上传成功 file_path : {str(file_path)}")

    try:
        import_file_to_vectorsdb(file_path,EMBEDDING_MODEL_PATH,EMBEDDING_MODEL,VECTOR_DB_PATH,'')
        return "文件上传成功，生成向量数据库成功！"
    except Exception as e:
        logging.error(f"An error occurred while saving the file: {str(e)}")
        return "上传文件失败，请重新上传！"


# 获取检索问题的 QA 对象
def _get_retrieval_qa(vertor_db,VECTOR_DB_PATH,LLM_NETWORK_KEY,LLM_NETWORK_URL,LLM_OLLAMA_MODEL,EMBEDDING_MODEL_PATH,EMBEDDING_MODEL):

    faiss_path = os.path.join(VECTOR_DB_PATH, str(vertor_db))

    llm = ChatOpenAI(
        api_key=LLM_NETWORK_KEY,  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url=LLM_NETWORK_URL,  # 填写DashScope base_url
        model=LLM_OLLAMA_MODEL
    )

    retriever = get_retriever(EMBEDDING_MODEL_PATH + '/' + EMBEDDING_MODEL, faiss_path)
    logging.info(f"get_retriever : {str(EMBEDDING_MODEL_PATH + '/' + EMBEDDING_MODEL)}")
    logging.info(f"get_retriever : {str(faiss_path)}")

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        verbose=True,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True
    )
    return qa


# 向量数据库聊天
def vertor_chat(vertor_db, question, chat_history):
    try:
        qa = _get_retrieval_qa(vertor_db)
        response = qa.invoke({"question": question, "chat_history": chat_history})
        return "S", response["answer"]
    except Exception as e:
        error_message = str(e)
        logging.error(error_message)
        return "E", error_message



