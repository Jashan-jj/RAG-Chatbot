# User interface using Gradio

from utils.LLM_utils import llm_response
import gradio as gr

def respond(message, chat_history):
    """
    Appends user message and a placeholder bot response ("Typing...") to the chat history.
    Then replaces it with the final response.

    Args:
        message (str): The user input message (query).
        chat_history (list): The list of previous chat message tuples.

    Returns:
        "" (str): Empty string for clearing input
        chat_history: Updated chat history list
    """
    response = "Typing..."
    chat_history.append((message, response))

    final_response = llm_response(message)   

    chat_history[-1] = (message, final_response)
    return "", chat_history

css = """
/* Dark radial background and fonts */
body {
    background: radial-gradient(circle at top left, #1f1f1f, #121212); 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}

/* Scale the entire chat block */
.zoom-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    transform: scale(1.125);
    transform-origin: top center;
}

/* Heading styles */
h1 {
    color: #ffffff;
    text-align: center;
    font-weight: 900;
    font-size: 2.5em;
    text-shadow: 0 2px 10px #4a90e2;
    margin-bottom: 0.5em;
}

/* User and bot message bubbles */
.gradio-chatbot-message.user {
    background-color: #f7a830;
    color: #ffffff;
    border-radius: 20px 20px 0 20px;
    padding: 12px 18px;
    font-weight: 600;
    max-width: 70%;
    animation: fadeInUp 0.3s ease-in;
}

.gradio-chatbot-message.bot {
    background-color: #4a90e2;
    color: #ffffff;
    border-radius: 20px 20px 20px 0;
    padding: 12px 18px;
    max-width: 70%;
    animation: fadeInUp 0.3s ease-in;
}

/* Chatbox container */
.gradio-chatbot {
    background: #222222;
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(255, 255, 255, 0.1);
    height: 375px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #666 #2a2a2a;
    padding: 15px;
    transition: all 0.3s ease-in-out;
}

/* Custom scrollbars */
.gradio-chatbot::-webkit-scrollbar {
    width: 8px;
}
.gradio-chatbot::-webkit-scrollbar-thumb {
    background: #4a90e2;
    border-radius: 4px;
}
.gradio-chatbot::-webkit-scrollbar-thumb:hover {
    background: #f7a830;
}

/* Input row (textbox + buttons) */
.input-row {
    display: flex;
    gap: 12px;
    justify-content: space-between;
    align-items: center;
    padding-top: 25px;
}

/* Textbox style */
.gr-textbox {
    flex-grow: 1;
    background: #2a2a2a;
    border-radius: 25px;
    border: 2px solid #f7a830;
    padding-left: 15px;
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    box-shadow: 0 3px 10px rgba(247, 168, 48, 0.3);
    transition: all 0.3s ease;
}
.gr-textbox:focus {
    border-color: #4a90e2;
    box-shadow: 0 3px 10px rgba(74, 144, 226, 0.5);
    outline: none;
}
.gr-textbox::placeholder {
    color: #aaa;
}

/* Clear button styling */
.gr-button {
    background-color: #4a90e2;
    color: #ffffff;
    border-radius: 25px;
    font-weight: 700;
    border: none;
    width: 120px;
    height: 40px;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.5);
    transition: 0.3s ease;
}
.gr-button:hover {
    background-color: #f7a830;
    color: #ffffff;
    box-shadow: 0 4px 15px rgba(247, 168, 48, 0.8);
    transform: scale(1.05);
    cursor: pointer;
}

/* Send button styling (circle with upper arrow) */
#send-button {
    width: 45px;
    height: 45px;
    background-color: #4a90e2;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    color: white;
    box-shadow: 0 4px 12px rgba(74, 144, 226, 0.5);
    transition: background-color 0.3s ease, transform 0.2s ease;
    padding: 0;
}
#send-button:hover {
    background-color: #f7a830;
    transform: scale(1.1);
    box-shadow: 0 4px 15px rgba(247, 168, 48, 0.8);
    cursor: pointer;
}
#send-button::before {
    content: '↑';
}

/* Container max width */
#container {
    max-width: 1000px;
    width: 100%;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
"""

# JavaScript to force dark mode in Gradio
js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

def launch_chatbot():
    """
    Launches the final chatbot interface.

    Returns:
        None
    """
    with gr.Blocks(css=css, js=js_func, title="RAG Chatbot") as demo:
        with gr.Column(elem_classes="zoom-wrapper"):
            with gr.Column(elem_id="container"):
                gr.Markdown("<h1>RAG Chatbot</h1>")
                
                # Chat window
                chatbot = gr.Chatbot(elem_classes="gradio-chatbot", show_copy_button=True)
                
                # Input + Buttons row
                with gr.Row(elem_classes="input-row"):
                    user_input = gr.Textbox(
                        placeholder="Type your message here...",
                        label="",
                        elem_classes="gr-textbox",
                        lines=1,
                        scale=8
                    )
                    with gr.Column(min_width=45, scale=0):
                        send_button = gr.Button("", elem_id="send-button")   # Send button
                    with gr.Column(min_width=120, scale=0):
                        clear = gr.Button("Clear Chat", elem_classes="gr-button")   # Clear Chat button 

        # Define button actions
        send_button.click(respond, [user_input, chatbot], [user_input, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

    demo.launch(share=True)
