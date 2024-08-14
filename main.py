import qyiutil_network
import panel as pn
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import param
import logging
import json  # 导入 json 模块以解析 JSON 响应

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class State(param.Parameterized):
    """
    保存程序的全局状态，包括本地机器人环境、目标文件夹、模型名称等。
    """
    instenv = qyiutil_network.networkbot()  # 初始化本地机器人环境
    target_folder = instenv.VECTOR_DB_PATH  # 设置目标文件夹路径
    model_name = instenv.LLM_OLLAMA_MODEL  # 设置LLM模型名称
    upload_file = param.Bytes()  # 参数化上传文件
    embeddingsmodel_name = instenv.EMBEDDING_MODEL_PATH + '/' + instenv.EMBEDDING_MODEL  # 设置嵌入模型路径

# 初始化机器人状态和聊天历史
state = State()
chat_history = []

# 获取向量数据库选项
vectordbs_options = qyiutil_network.get_vectordbs(state.instenv.VECTOR_DB_PATH)

# 创建向量数据库选择器
chain_type_input = pn.widgets.RadioButtonGroup(
    name="本地向量数据库名称",
    options=vectordbs_options,
    orientation="vertical",
    sizing_mode="stretch_width",
    button_type="primary",
    button_style="outline",
)

# 定义输入文本框和文件输入框
text_input = pn.widgets.TextInput(name="", placeholder="输入你想提问的问题！")
file_input = pn.widgets.FileInput(name="上传文件", accept=".pdf,.csv,.txt")


# 定义获取机器人回应的函数
def _get_response(contents):
    """
    获取机器人回应，根据是否选择了向量数据库来决定回应的方式。
    """
    logging.info("获取机器人回应")
    logging.info(f"chain_type_input.value: {str(chain_type_input.value)}")
    logging.info(f"contents: {str(contents)}")

    if chain_type_input.value is None:
        llm = ChatOpenAI(
            api_key= state.instenv.LLM_NETWORK_KEY, # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url=state.instenv.LLM_NETWORK_URL, # 填写DashScope base_url
            model=state.instenv.LLM_OLLAMA_MODEL
            )
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=contents),
        ]
        response = llm.invoke(messages)
        content_value = response.json(ensure_ascii=False)
        response = json.loads(content_value)
        response = response['content']
        chat_history.append((contents, response))
    else:
        qa = qyiutil_network._get_retrieval_qa(chain_type_input.value, state.instenv.VECTOR_DB_PATH,state.instenv.LLM_NETWORK_KEY,
                                               state.instenv.LLM_NETWORK_URL, state.instenv.LLM_OLLAMA_MODEL, state.instenv.EMBEDDING_MODEL_PATH,
                                               state.instenv.EMBEDDING_MODEL)

        response = qa.invoke({"question": contents, "chat_history": chat_history})
        chat_history.append((contents, response["answer"]))
        response = response["answer"]

    return response, []

# 定义响应用户输入的函数
async def respond(contents, user, chat_interface):
    """
    响应用户输入的函数，处理用户的文本输入和文件上传。
    """
    if chat_interface.active == 0:
        response, _ = _get_response(contents)
        yield {"user": "Qyi", "avatar": "🤖️", "object": response}
    elif chat_interface.active == 1:
        chat_interface.active = 0
        yield {"user": "Qyi", "avatar": "🤖️", "object": "文件正在处理中，请稍后...."}

        upload_message = qyiutil_network._save_file(contents, chat_interface, state.instenv.FILE_UPLOAD_PATH,
                                            state.instenv.VECTOR_DB_PATH, state.instenv.EMBEDDING_MODEL_PATH,
                                            state.instenv.EMBEDDING_MODEL, '')
        # 更新向量数据库选项
        chain_type_input.options = qyiutil_network.get_vectordbs(state.instenv.VECTOR_DB_PATH)
        yield {"user": "Qyi", "avatar": "🤖️", "object": "文件上传完成," + str(upload_message)}

# 创建聊天界面
chat_interface = pn.chat.ChatInterface(
    widgets=[text_input, file_input],
    callback=respond,
    callback_exception='verbose',
    sizing_mode="stretch_width",
    renderers=pn.pane.Perspective,
    show_rerun=False,
    show_undo=False,
    show_clear=True,
)

# 发送欢迎消息
chat_interface.send(
    "QyiAI-网络版 1.0",
    user="system",
    respond=False
)

# 获取当前模型的描述
model_desc = '模型：' + state.model_name

# 创建 Bootstrap 模板
template = pn.template.BootstrapTemplate(
    title="QyiAi-网络版-V1.0",
    sidebar=[model_desc, chain_type_input],
    main=[chat_interface],
)

# 显示模板
template.servable()
template.show()

logging.info("QyiAI-知识库网络版启动成功")
