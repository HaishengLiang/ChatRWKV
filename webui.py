########################################################################################################
# The RWKV Chat WebUI
# Made by knightlhs@qq.com
########################################################################################################

import os, copy, types, gc, sys
import numpy as np
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass
np.set_printoptions(precision=4, suppress=True, linewidth=200)
args = types.SimpleNamespace()

print('\n\nChatRWKV project: https://github.com/BlinkDL/ChatRWKV')

########################################################################################################

args.RUN_DEVICE = "cuda"  # cuda // cpu
# fp16 (good for GPU, does NOT support CPU) // fp32 (good for CPU) // bf16 (worse accuracy, supports CPU)
args.FLOAT_MODE = "fp16"

os.environ["RWKV_JIT_ON"] = '1' # '1' or '0', please use torch 1.13+ and benchmark speed

QA_PROMPT = False # True: Q & A prompt // False: User & Bot prompt
# 中文问答设置QA_PROMPT=True（只能问答，问答效果更好，但不能闲聊） 中文聊天设置QA_PROMPT=False（可以闲聊，但需要大模型才适合闲聊）

# Download RWKV-4 models from https://huggingface.co/BlinkDL (don't use Instruct-test models unless you use their prompt templates)


args.ctx_len = 1024

CHAT_LEN_SHORT = 40
CHAT_LEN_LONG = 150
FREE_GEN_LEN = 200

GEN_TEMP = 1.0
GEN_TOP_P = 0.85

AVOID_REPEAT = '，。：？！'

########################################################################################################
########################################################################################################
import torch

# please tune these (test True/False for all of them). can significantly improve speed.
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
# torch._C._jit_override_can_fuse_on_cpu(True)
# torch._C._jit_override_can_fuse_on_gpu(True)
# torch._C._jit_set_texpr_fuser_enabled(False)
# torch._C._jit_set_nvfuser_enabled(False)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from src.model_run import RWKV_RNN
from src.utils import TOKENIZER
tokenizer = TOKENIZER("20B_tokenizer.json")

args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

# Load Model
# print(f'Loading model - {MODEL_NAME}')
# model = RWKV_RNN(args)
model = None
model_tokens = []
model_state = None

AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = tokenizer.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd

########################################################################################################
def run_rnn(tokens, newline_adj = 0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens
    out, model_state = model.forward(tokens, model_state)

    # print(f'### model ###\n{tokens}\n[{tokenizer.decode(model_tokens)}]')

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj # adjust \n probability
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999
    return out

all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, model_state
    n = f'{name}_{srv}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

########################################################################################################
def start_model(run_mode, language, model_size):
    global model
    if model:
        print('model is started!')
    else:
        os.environ["RWKV_RUN_DEVICE"] = run_mode
        print(f'\nLoading ChatRWKV - {language} - {args.RUN_DEVICE} - {args.FLOAT_MODE} - QA_PROMPT {QA_PROMPT}')
        if run_mode == 'GPU':
            args.RUN_DEVICE = "cuda"  # cuda // cpu
            args.FLOAT_MODE = "fp16"
        else:
            args.RUN_DEVICE = "cpu"  # cuda // cpu
            args.FLOAT_MODE = "fp32"

        if language == 'English':
            args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-14b/RWKV-4-Pile-14B-20230213-8019'
            # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-20221115-8047'
            # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096'
            # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040'
            # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-430m/RWKV-4-Pile-430M-20220808-8066'
            # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-169m/RWKV-4-Pile-169M-20220807-8023'
            # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-340'
            # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/14b-run1/rwkv-6210'

        elif language == 'Chinese': # testNovel系列是网文模型，请只用 +gen 指令续写。test4 系列可以问答（只用了小中文语料微调，纯属娱乐）
            # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-7b/RWKV-4-Pile-7B-EngChn-testNovel-441-ctx2048-20230217'
            args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-EngChn-testNovel-1136-ctx2048-20230218'
            # args.MODEL_NAME = '/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-EngChn-testNovel-671-ctx2048-20230216'
            # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/7-run1z/rwkv-461'
            # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/3-run1z/rwkv-711'
            # args.MODEL_NAME = '/fsx/BlinkDL/CODE/_PUBLIC_/RWKV-LM/RWKV-v4neo/1.5-run1z/rwkv-671'
        MODEL_NAME = args.MODEL_NAME

        # load model
        model = RWKV_RNN(args)
        gc.collect()
        torch.cuda.empty_cache()
        print(f'\nReady - {language} {args.RUN_DEVICE} {args.FLOAT_MODE} QA_PROMPT={QA_PROMPT} {args.MODEL_NAME}\n')
    return gr.Button.update(visible=False), gr.Button.update(visible=True), gr.Radio.update(interactive=False), gr.Radio.update(interactive=False), gr.Radio.update(interactive=False)

def stop_model():
    global model
    print(f'\nStop ChatRWKV')
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    return gr.Button.update(visible=True), gr.Button.update(visible=False), gr.Radio.update(interactive=True), gr.Radio.update(interactive=True), gr.Radio.update(interactive=True)

########################################################################################################

import gradio as gr

# 标题
title = "ChatRWKV WebUI Demo"

description = "ChatRWKV project: https://github.com/BlinkDL/ChatRWKV"

all_text = ''
isSelected = False

def decode_tokens(tokens):
    global model_tokens, model_state
    # generate save state
    generate_text = ''

    srv = 'dummy_server'

    # decode
    x_temp = GEN_TEMP
    x_top_p = GEN_TOP_P

    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0


    print(f'Decode - {x_temp} {x_top_p} {FREE_GEN_LEN}\n')

    begin = len(model_tokens)
    out_last = begin
    out = tokens
    for i in range(FREE_GEN_LEN+100):
        token = tokenizer.sample_logits(
            out,
            model_tokens,
            args.ctx_len,
            temperature=x_temp,
            top_p=x_top_p,
        )
        out = run_rnn([token])
        xxx = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx: # avoid utf-8 display issues
            # print(xxx, end='', flush=True)
            generate_text = generate_text + xxx
            out_last = begin + i + 1
            if i >= FREE_GEN_LEN:
                break
        
        # print('\n')
        # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
    
    save_all_stat(srv, 'gen_1', out)
    print(f'Generate Text - {generate_text}\n')
    return generate_text

# 内容生成函数
def generate(input = ''):
    global model_tokens, model_state, first_generate
    if input == '':
        print('Error: please say something')
        return
    if model == None:
        print('Error: please start model first')
        return

    # begin generate
    srv = 'dummy_server'
    # if len(msg) > 1000:
    #     reply_msg('your message is too long (max 1000 tokens)')
    #     return
    msg = input.replace('\\n','\n').strip()
    new = '\n' + input
    # print(f'### prompt ###\n[{new}]')
    model_state = None
    model_tokens = []
    out = run_rnn(tokenizer.encode(new))
    save_all_stat(srv, 'gen_0', out)
    generate_text = decode_tokens(out)
    first_generate = True
    return input + generate_text

# 继续生成
def generate_more():
    global isSelected
    generate_text = ''
    isSelected = False
    # begin generate
    srv = 'dummy_server'
    try:
        out = load_all_stat(srv, 'gen_1')
        save_all_stat(srv, 'gen_0', out)
    except:
        return f'find some error'
    generate_text = decode_tokens(out)
    # gen_output, regen_button, select_button, more_button
    return generate_text, gr.Button.update(visible=True), gr.Button.update(visible=True), gr.Button.update(visible=False)

# 重新生成
def generate_retry(input):
    global all_text, isSelected, first_generate
    if isSelected:
        return ''
    generate_text = ''
    # begin generate
    srv = 'dummy_server'
    try:
        out = load_all_stat(srv, 'gen_0')
    except:
        return f'find some error'
    generate_text = decode_tokens(out)
    if first_generate == True:
        return input + generate_text
    else:
        return generate_text

def select_generation(gen_content):
    global all_text, isSelected, first_generate
    if gen_content == '' or isSelected:
        return all_text
    all_text = all_text + gen_content
    isSelected = True
    first_generate = False
    return all_text, gr.Button.update(visible=False), gr.Button.update(visible=False), gr.Button.update(visible=True)

def clean_generation():
    global all_text, model_state, model_tokens
    all_text = ''
    model_state = None
    model_tokens = []
    return '', ''

# 对话处理
def chat_demo(chat_type, chat_input, user_name, bot_name, history=[]):
    print(f'{chat_type}-{chat_input}-{user_name}-{bot_name}')
    reply = f'output - {chat_input}'
    return [(chat_input, reply)], history

# 参数调节回调
def free_gen_len_change(len):
    global FREE_GEN_LEN
    FREE_GEN_LEN = int(len)

def topp_change(topp):
    global GEN_TOP_P
    GEN_TOP_P = topp

# 生成内容样例
gen_examples = [
    ["以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\n第一章", 0.85],
    ["我已经老了。有一天，在一个公共大厅里，一个男子向我走来，对我说：“我认识你，我曾经爱过你，那时你还很年轻，但与那时相比，我更爱你现在饱经风霜的容颜。", 0.75],
    ["钱塘江浩浩江水，日日夜夜无穷无休地从临安牛家村边绕过，东流入海。江畔一排数十株乌桕树，叶子似火烧般红，正是八月天时。村前村后的野草刚起始变黄，一抹斜阳映照之下，更增了几分萧索。两株大松树下围着一堆村民，男男女女和十几个小孩，正自聚精会神地听着一个瘦削的老者说话。", 0.70],
]

chat_examples = [

]

# 全局变量

# 全局样式
css = """
.gr-button-danger {
    background-color: #ff4a4ac2 !important;
    color: #fff !important;
}
div.output-class {
    word-break: break-all;
    min-height: 200px;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    white-space: pre-line;
    overflow: scroll;
    max-height: 400px;
}
"""

with gr.Blocks(css=css) as app:  
    # 功能描述
    gr.Markdown(
        f'''
            # {title}
            {description}
        '''
    ) 
    # 模型管理板块
    with gr.Row():
        with gr.Column():
            run_mode = gr.Radio(label="Device", type="value", choices=["GPU", "CPU"], value="GPU", elem_id="run_mode")
            language = gr.Radio(label="Language", choices=["Chinese", "English"], value="Chinese", elem_id="language")
            model_size = gr.Radio(label="Model Size", choices=["1B5", "3B", "7B", "14B"], value="3B")
            
            with gr.Row():
                with gr.Column():
                    start_button = gr.Button("Start Model", variant="primary", visible=True)
                    stop_button = gr.Button("Stop Model", variant="danger", visible=False)
                    
        start_button.click(start_model, inputs=[run_mode, language, model_size], outputs=[start_button, stop_button, run_mode, language, model_size])
        stop_button.click(stop_model, inputs=None, outputs=[start_button, stop_button, run_mode, language, model_size])
        
    # 实例展示板块
    with gr.Tabs():
        # 文章生成
        with gr.TabItem("Article Generate", elem_id="generate-tab"):
            with gr.Row():
                # 参数配置面板
                with gr.Column():
                    text_input = gr.TextArea(label="Article begin with: ")
                    gen_temp = gr.Slider(label="GEN-TEMP", value=GEN_TEMP, minimum=0.2, maximum=5)
                    free_gen_len = gr.Slider(label="FREE-GEN-LEN", value=FREE_GEN_LEN, minimum=100, maximum=1200, step=100)
                    top_p = gr.Slider(label="TOP-P", value=GEN_TOP_P, minimum=0, maximum=1.00, step=0.01)
                    with gr.Row():
                        with gr.Column():
                            gen_button = gr.Button("Generate", variant="primary", elem_id="gen_button")
                        with gr.Column():
                            clear_button = gr.Button("Clean", variant="danger")
                    
                    
                # 生成结果面板
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            gen_output = gr.TextArea(label="Generate Result")
                            select_button = gr.Button("Select", variant="primary", elem_id="select_button")
                    with gr.Row():
                        with gr.Column():
                            more_button = gr.Button("More", variant="secondary", elem_id="gen_button", visible=False)
                        with gr.Column():
                            regen_button = gr.Button("ReTry", variant="secondary", elem_id="regen_button")
                    with gr.Row():
                        result_label = gr.Label(label="Full Article Text")
                
                # 交互事件处理
                select_button.click(select_generation, inputs=gen_output, outputs=[result_label, regen_button, select_button, more_button])
                clear_button.click(clean_generation, outputs=[gen_output, result_label])

            # 动作事件处理
            free_gen_len.change(free_gen_len_change, inputs=free_gen_len)
            top_p.change(topp_change, inputs=top_p)
            gen_button.click(generate, inputs=text_input, outputs=gen_output)
            # 
            more_button.click(generate_more, inputs=None, outputs=[gen_output, regen_button, select_button, more_button])
            regen_button.click(generate_retry, inputs=text_input, outputs=gen_output)
            with gr.Row():
                gr.Examples(label="Examples", examples=gen_examples, inputs=[text_input, top_p])
        
        # 对话与问答
        with gr.TabItem("Chat And Q&A", elem_id="chat-tab"):  
            with gr.Row():
                with gr.Column():
                    chat_radio = gr.Radio(label="Please select mode for chat: ", type="index", choices=["Chat", "Q&A"], value="Chat")
                    chat_input = gr.Text(label="Input", value="", elem_id="chat-input")
                    state = gr.State([])
                    user_name = gr.Text(label="User Name", value="User", elem_id="user")
                    bot_name = gr.Text(label="Bot Name", value="Bot", interactive=False, elem_id="bot")
                    chat_button = gr.Button("Submit", variant="primary", interactive=False, elem_id="chat-button")
                with gr.Column():
                    chatbot = gr.Chatbot()
                    # .style(color_map=[])
                chat_button.click(fn=chat_demo, inputs=[chat_radio, chat_input, user_name, bot_name, state], outputs=[chatbot, state])

# main
if __name__ == "__main__":
    app.launch()
    