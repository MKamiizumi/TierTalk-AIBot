import json
import random
import io
import uvicorn
import logging
from enum import Enum
from typing import Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from google.cloud import speech
from pydub import AudioSegment
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langserve import add_routes,CustomUserType
from langchain.pydantic_v1 import Field
from langchain_core.messages import AIMessage,BaseMessage,HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

# Load environment variables from .env file
load_dotenv()

# basic
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
logging.basicConfig(level=logging.DEBUG) # debug logging

# Function A. tier choice
class Level(str, Enum):
    cet4="cet4"     
    cet6="cet6"     
    junior="junior"  
    sat="sat"
    senior="senior"   
    toefl="toefl"    

class ChatMessage(CustomUserType):
    input: str = Field(
        None,
        examples=["hello"],
        description="The user's input",
        extra={"widget": {"type": "text"}},
    )
    level: Level = Field(
        ...,
        description="The level of the user",
    )

def load_vocabulary(level: str) -> list:
    file_mapping = {
        "cet4": './data/cet4.json',
        "cet6": './data/cet6.json',
        "junior": './data/junior.json',
        "sat": './data/sat.json',
        "senior": './data/senior.json',
        "toefl": './data/toefl.json',
    }

    file_path = file_mapping.get(level, file_mapping.get('cet4'))

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as file:
            data = json.load(file)

    return [entry['word'] for entry in data]

def get_shuffled_vocab(vocab_list):
    shuffled = vocab_list.copy()
    random.shuffle(shuffled)
    return ", ".join(shuffled[:20])


# Function B: Voice Input
# 一、提供 index.html 页面
@app.get("/")
async def index():
    return FileResponse('static/index.html')

# 二、处理音频

# 1. 仅处理音频并转录，不保存文件
@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), level: str = Form(...)):
    logging.debug(f"Received file: {file.filename}, content type: {file.content_type}")
    logging.debug(f"User level: {level}")

    # 检查是否为wav文件，不是则报错
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Invalid audio file type")

    
    audio_content = await file.read()  # 读取音频内容
    logging.debug(f"Audio content size: {len(audio_content)} bytes")

    # 如果文件大小为 0，返回错误
    if len(audio_content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    # 调用 Google Speech-to-Text API 转录音频
    transcript = transcribe_audio(audio_content)

    # 调用 GPT 模型并传递用户级别
    gpt_response = invoke_gpt_model(transcript, level)
    
    return {'message': 'Audio uploaded and transcribed successfully', 'transcript': transcript}

# 2. Transcribe_audio函数：调用 Google Cloud Speech-to-Text API 进行转录
def transcribe_audio(audio_content):
    client = speech.SpeechClient()

    # 将音频内容加载到 pydub 的 AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(audio_content))

    # Print and Test
    original_frame_rate = audio.frame_rate
    print(f"Original sample rate: {original_frame_rate} Hz")

    # 将音频采样率转换为 16000 Hz
    audio = audio.set_frame_rate(16000)

    # 将转换后的音频保存到字节流
    buffer = io.BytesIO()
    audio.export(buffer, format="wav", codec="pcm_s16le")
    buffer.seek(0)

    # send to Google API
    converted_audio_content = buffer.read() # Read Buffer content

    audio = speech.RecognitionAudio(content = converted_audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    transcript = ''.join([result.alternatives[0].transcript for result in response.results])
    
    return transcript

# 3. 用text调用 GPT 模型生成回复（to be modified)
def invoke_gpt_model(transcribed_text, user_level):
    # 将 transcribed_text 和 level 封装成 ChatMessage 对象
    input_message = ChatMessage(input=transcribed_text, level=user_level)
    
    # 通过 chat_message_bot 生成消息链
    messages = chat_message_bot(input_message)
    
    # 运行 GPT 模型，生成回复
    gpt_response = model(messages)
    
    # 提取 GPT 的最终回复
    return output_message(gpt_response)

# Main: Interaction with GPT（待修改）
def get_prompt(vocab: str) -> str:
    return f"You are an English tutor. Use the following vocabulary in your responses: {vocab}"

def chat_message_bot(input: ChatMessage) -> str :
    user_input = input.input
    vocab_list = load_vocabulary(input.level.value)
    random_vocab = get_shuffled_vocab(vocab_list)
    prompt = get_prompt(random_vocab)

    messages = [
        AIMessage(content=prompt),
        HumanMessage(content=user_input)
    ]
    return messages

def output_message(messages: Tuple[BaseMessage, BaseMessage]) -> str:
    return messages.content

model = ChatOpenAI(model="gpt-4o-mini", request_timeout=60)

# Add Routes
add_routes(
    app,
    RunnableParallel({"answer": chat_message_bot | model | RunnableLambda(output_message)}).with_types(input_type=ChatMessage),
    config_keys=["configurable"],
    path="",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000, log_level="debug")


