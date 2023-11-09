# Welcome to Team Tonic's MultiMed

from gradio_client import Client
import os
import numpy as np
import base64
import gradio as gr
import requests
import json
import dotenv
from scipy.io.wavfile import write
import PIL
from openai import OpenAI
import time
dotenv.load_dotenv()

seamless_client = Client("facebook/seamless_m4t")
HuggingFace_Token = os.getenv("HuggingFace_Token")

def check_hallucination(assertion,citation):
    API_URL = "https://api-inference.huggingface.co/models/vectara/hallucination_evaluation_model"
    headers = {"Authorization": f"Bearer {HuggingFace_Token}"}
    payload = {"inputs" : f"{assertion} [SEP] {citation}"}

    response = requests.post(API_URL, headers=headers, json=payload,timeout=120)
    output = response.json()
    output = output[0][0]["score"]

    return f"**hullicination score:** {output}"



def process_speech(input_language, audio_input):
    """
    processing sound using seamless_m4t
    """
    if audio_input is None :
        return "no audio or audio did not save yet \nplease try again ! "
    print(f"audio : {audio_input}")
    print(f"audio type : {type(audio_input)}")
    out = seamless_client.predict(
        "S2TT",
        "file",
        None,
        audio_input, #audio_name
        "",
        input_language,# source language
        input_language,# target language
        api_name="/run",
    )
    out = out[1] # get the text
    try :
        return f"{out}"
    except Exception as e :
        return f"{e}"




def process_image(image) : 
    img_name = f"{np.random.randint(0, 100)}.jpg"
    PIL.Image.fromarray(image.astype('uint8'), 'RGB').save(img_name)
    image = open(img_name, "rb").read()
    base64_image = base64_image = base64.b64encode(image).decode('utf-8')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    # oai_org = os.getenv('OAI_ORG')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "You are clinical consultant discussion training cases with students at TonicUniversity. Assess and describe the photo in minute detail. Explain why each area or item in the photograph would be inappropriate to describe if required. Pay attention to anatomy, symptoms and remedies. Propose a course of action based on your assessment. Exclude any other commentary:"
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 1200
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    try :
        out = response.json()
        out = out["choices"][0]["message"]["content"]

        return out
    except Exception as e :
        return f"{e}"


def query_vectara(text):
    user_message = text

    # Read authentication parameters from the .env file
    CUSTOMER_ID = os.getenv('CUSTOMER_ID')
    CORPUS_ID = os.getenv('CORPUS_ID')
    API_KEY = os.getenv('API_KEY')

    # Define the headers
    api_key_header = {
        "customer-id": CUSTOMER_ID,
        "x-api-key": API_KEY
    }

    # Define the request body in the structure provided in the example
    request_body = {
        "query": [
            {
                "query": user_message,
                "queryContext": "",
                "start": 1,
                "numResults": 50,
                "contextConfig": {
                    "charsBefore": 0,
                    "charsAfter": 0,
                    "sentencesBefore": 2,
                    "sentencesAfter": 2,
                    "startTag": "%START_SNIPPET%",
                    "endTag": "%END_SNIPPET%",
                },
                "rerankingConfig": {
                    "rerankerId": 272725718,
                    "mmrConfig": {
                        "diversityBias": 0.35
                    }
                },
                "corpusKey": [
                    {
                        "customerId": CUSTOMER_ID,
                        "corpusId": CORPUS_ID,
                        "semantics": 0,
                        "metadataFilter": "",
                        "lexicalInterpolationConfig": {
                            "lambda": 0
                        },
                        "dim": []
                    }
                ],
                "summary": [
                    {
                        "maxSummarizedResults": 5,
                        "responseLang": "auto",
                        "summarizerPromptName": "vectara-summary-ext-v1.2.0"
                    }
                ]
            }
        ]
    }

    # Make the API request using Gradio
    response = requests.post(
        "https://api.vectara.io/v1/query",
        json=request_body,  # Use json to automatically serialize the request body
        verify=True,
        headers=api_key_header
    )

    if response.status_code == 200:
        query_data = response.json()
        if query_data:
            sources_info = []

            # Extract the summary.
            summary = query_data['responseSet'][0]['summary'][0]['text']

            # Iterate over all response sets
            for response_set in query_data.get('responseSet', []):
                # Extract sources
                # Limit to top 5 sources.
                for source in response_set.get('response', [])[:5]:
                    source_metadata = source.get('metadata', [])
                    source_info = {}

                    for metadata in source_metadata:
                        metadata_name = metadata.get('name', '')
                        metadata_value = metadata.get('value', '')

                        if metadata_name == 'title':
                            source_info['title'] = metadata_value
                        elif metadata_name == 'author':
                            source_info['author'] = metadata_value
                        elif metadata_name == 'pageNumber':
                            source_info['page number'] = metadata_value

                    if source_info:
                        sources_info.append(source_info)

            result = {"summary": summary, "sources": sources_info}
            return f"{json.dumps(result, indent=2)}"
        else:
            return "No data found in the response."
    else:
        return f"Error: {response.status_code}"


def convert_to_markdown(vectara_response_json):
    vectara_response = json.loads(vectara_response_json)
    if vectara_response:
        summary = vectara_response.get('summary', 'No summary available')
        sources_info = vectara_response.get('sources', [])

        # Format the summary as Markdown
        markdown_summary = f' {summary}\n\n'

        # Format the sources as a numbered list
        markdown_sources = ""
        for i, source_info in enumerate(sources_info):
            author = source_info.get('author', 'Unknown author')
            title = source_info.get('title', 'Unknown title')
            page_number = source_info.get('page number', 'Unknown page number')
            markdown_sources += f"{i+1}. {title} by {author}, Page {page_number}\n"

        return f"{markdown_summary}**Sources:**\n{markdown_sources}"
    else:
        return "No data found in the response."
# Main function to handle the Gradio interface logic

def process_summary_with_openai(summary):
    """
    This function takes a summary text as input and processes it with OpenAI's GPT model.
    """
    try:
        # Ensure that the OpenAI client is properly initialized
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create the prompt for OpenAI's completion
        prompt = "You are clinical consultant discussion training cases with students at TonicUniversity. You will recieve a summary assessment. Assess and describe the proper options in minute detail. Propose a course of action based on your assessment. Exclude any other commentary:"
        
        # Call the OpenAI API with the prompt and the summary
        completion = client.chat.completions.create(
            model="gpt-4-1106-preview",  # Make sure to use the correct model name
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": summary}
            ]
        )
        
        # Extract the content from the completion
        final_summary = completion.choices[0].message.content
        return final_summary
    except Exception as e:
        return str(e)
        

def process_and_query(text=None):
    try:
        # augment the prompt before feeding it to vectara
        text = "the user asks the following to his health adviser " + text
        # If an image is provided, process it with OpenAI and use the response as the text query for Vectara
        # if image is not None:
        #     text = process_image(image)
        #     return "**Summary:** "+text
        # if audio is not None:
        #     text = process_speech(audio)
        #     # augment the prompt before feeding it to vectara
        #     text = "the user asks the following to his health adviser " + text
            

        
        # Use the text to query Vectara
        vectara_response_json = query_vectara(text)
        
        # Convert the Vectara response to Markdown
        markdown_output = convert_to_markdown(vectara_response_json)
        
        # Process the summary with OpenAI
        final_response = process_summary_with_openai(markdown_output)
        
        # Return the processed summary along with the full output
        return f"**Summary**: {final_response}\n\n**Full output**:\n{markdown_output}"
    except Exception as e:
        return str(e)

        completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": markdown_output_final}
          ]
        )
        final_response= completion.choices[0].message.content
        return f"**Summary**: {final_response}\n\n**Full output**:\n{markdown_output}"
    except Exception as e:
        return str(e)


# Define the Gradio interface
# iface = gr.Interface(
#     fn=process_and_query,
#     inputs=[
#         gr.Textbox(label="Input Text"),
#         gr.Image(label="Upload Image"),
#         gr.Audio(label="talk in french",
#                  sources=["microphone"]),
#     ],
#     outputs=[gr.Markdown(label="Output Text")],
#     title="👋🏻Welcome to ⚕🗣️😷MultiMed - Access Chat ⚕🗣️😷",
#     description='''
#             ### How To Use ⚕🗣️😷MultiMed⚕: 
#             #### 🗣️📝Interact with ⚕🗣️😷MultiMed⚕ in any language using audio or text!
#             #### 🗣️📝 This is an educational and accessible conversational tool to improve wellness and sanitation in support of public health. 
#             #### 📚🌟💼 The knowledge base is composed of publicly available medical and health sources in multiple languages. We also used [Kelvalya/MedAware](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) that we processed and converted to HTML. The quality of the answers depends on the quality of the dataset, so if you want to see some data represented here, do [get in touch](https://discord.gg/GWpVpekp). You can also use 😷MultiMed⚕️ on your own data & in your own way by cloning this space. 🧬🔬🔍 Simply click here: <a style="display:inline-block" href="https://huggingface.co/spaces/TeamTonic/MultiMed?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></h3>
#             #### Join us : 🌟TeamTonic🌟 is always making cool demos! Join our active builder's🛠️community on 👻Discord: [Discord](https://discord.gg/GWpVpekp) On 🤗Huggingface: [TeamTonic](https://huggingface.co/TeamTonic) & [MultiTransformer](https://huggingface.co/MultiTransformer) On 🌐Github: [Polytonic](https://github.com/tonic-ai) & contribute to 🌟 [PolyGPT](https://github.com/tonic-ai/polygpt-alpha)"
#             ''',
#     theme='ParityError/Anime',
#     examples=[
#         ["What is the proper treatment for buccal herpes?"],
#         ["Male, 40 presenting with swollen glands and a rash"],
#         ["How does cellular metabolism work TCA cycle"],
#         ["What special care must be provided to children with chicken pox?"],
#         ["When and how often should I wash my hands ?"],
#         ["بکل ہرپس کا صحیح علاج کیا ہے؟"],
#         ["구강 헤르페스의 적절한 치료법은 무엇입니까?"],
#         ["Je, ni matibabu gani sahihi kwa herpes ya buccal?"],
#     ],
# )

welcome_message = """
# 👋🏻Welcome to ⚕🗣️😷MultiMed - Access Chat ⚕🗣️😷
### How To Use ⚕🗣️😷MultiMed⚕: 
#### 🗣️📝Interact with ⚕🗣️😷MultiMed⚕ in any language using audio or text!
#### 🗣️📝 This is an educational and accessible conversational tool to improve wellness and sanitation in support of public health. 
#### 📚🌟💼 The knowledge base is composed of publicly available medical and health sources in multiple languages. We also used [Kelvalya/MedAware](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) that we processed and converted to HTML. The quality of the answers depends on the quality of the dataset, so if you want to see some data represented here, do [get in touch](https://discord.gg/GWpVpekp). You can also use 😷MultiMed⚕️ on your own data & in your own way by cloning this space. 🧬🔬🔍 Simply click here: <a style="display:inline-block" href="https://huggingface.co/spaces/TeamTonic/MultiMed?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></h3>
#### Join us : 🌟TeamTonic🌟 is always making cool demos! Join our active builder's🛠️community on 👻Discord: [Discord](https://discord.gg/GWpVpekp) On 🤗Huggingface: [TeamTonic](https://huggingface.co/TeamTonic) & [MultiTransformer](https://huggingface.co/MultiTransformer) On 🌐Github: [Polytonic](https://github.com/tonic-ai) & contribute to 🌟 [PolyGPT](https://github.com/tonic-ai/polygpt-alpha)"             
"""


languages = [
    "Afrikaans",
    "Amharic",
    "Modern Standard Arabic",
    "Moroccan Arabic",
    "Egyptian Arabic",
    "Assamese",
    "Asturian",
    "North Azerbaijani",
    "Belarusian",
    "Bengali",
    "Bosnian",
    "Bulgarian",
    "Catalan",
    "Cebuano",
    "Czech",
    "Central Kurdish",
    "Mandarin Chinese",
    "Welsh",
    "Danish",
    "German",
    "Greek",
    "English",
    "Estonian",
    "Basque",
    "Finnish",
    "French",
    "West Central Oromo",
    "Irish",
    "Galician",
    "Gujarati",
    "Hebrew",
    "Hindi",
    "Croatian",
    "Hungarian",
    "Armenian",
    "Igbo",
    "Indonesian",
    "Icelandic",
    "Italian",
    "Javanese",
    "Japanese",
    "Kamba",
    "Kannada",
    "Georgian",
    "Kazakh",
    "Kabuverdianu",
    "Halh Mongolian",
    "Khmer",
    "Kyrgyz",
    "Korean",
    "Lao",
    "Lithuanian",
    "Luxembourgish",
    "Ganda",
    "Luo",
    "Standard Latvian",
    "Maithili",
    "Malayalam",
    "Marathi",
    "Macedonian",
    "Maltese",
    "Meitei",
    "Burmese",
    "Dutch",
    "Norwegian Nynorsk",
    "Norwegian Bokmål",
    "Nepali",
    "Nyanja",
    "Occitan",
    "Odia",
    "Punjabi",
    "Southern Pashto",
    "Western Persian",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Slovak",
    "Slovenian",
    "Shona",
    "Sindhi",
    "Somali",
    "Spanish",
    "Serbian",
    "Swedish",
    "Swahili",
    "Tamil",
    "Telugu",
    "Tajik",
    "Tagalog",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Northern Uzbek",
    "Vietnamese",
    "Xhosa",
    "Yoruba",
    "Cantonese",
    "Colloquial Malay",
    "Standard Malay",
    "Zulu"
]


with gr.Blocks(theme='ParityError/Anime') as iface : 
    gr.Markdown(welcome_message)
    with gr.Tab("text summarization"):
        text_input = gr.Textbox(label="input text",lines=5)
        text_output = gr.Markdown(label="output text")
        text_button = gr.Button("process text")
        gr.Examples(["my skin is swollen, and i have a high fever"],inputs=[text_input])
    with gr.Tab("image identification"):
        image_input = gr.Image(label="upload image")
        image_output = gr.Markdown(label="output text")
        image_button = gr.Button("process image")
        image_button.click(process_image, inputs=image_input, outputs=image_output)
        gr.Examples(["sick person.jpeg"],inputs=[image_input])
    with gr.Tab("speech to text"):
        input_language = gr.Dropdown(languages, label="select the language",value="English",interactive=True)
        audio_input = gr.Audio(label="speak",type="filepath",sources="microphone")
        audio_output = gr.Markdown(label="output text")
        audio_button = gr.Button("process audio")
        audio_button.click(process_speech, inputs=[input_language,audio_input], outputs=audio_output)
        gr.Examples([["English","sample_input.mp3"]],inputs=[input_language,audio_input])
    with gr.Tab("hallucination check"):
        assertion = gr.Textbox(label="assertion")
        citation =  gr.Textbox(label="citation text")
        hullucination_output = gr.Markdown(label="output text")
        audio_button = gr.Button("check hallucination")
        gr.Examples([["i am drunk","sarah is pregnant"]],inputs=[assertion,citation])
    text_button.click(process_and_query, inputs=text_input, outputs=text_output)
    audio_button.click(check_hallucination,inputs=[assertion,citation],outputs=hullucination_output)
    



iface.queue().launch(show_error=True,debug=True)
