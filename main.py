# app.py
import requests
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# New GenAI imports (Gemini)
from google import genai
from google.genai import types  # for structured Content/Part objects

app = Flask(__name__)
CORS(app)

# Globals to hold keys and the GenAI client
api_key = None
hf_api_token = None
genai_client = None  # will hold genai.Client(api_key=...)

# Optional: change this to a model you have access to
# Examples: "gemini-pro", "gemini-2.5-flash"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


@app.route('/api/set-api-key', methods=['POST'])
def set_api_key():
    global api_key, hf_api_token, genai_client
    data = request.json

    api_key = data.get('api_key', '').strip()
    hf_api_token = data.get('hf_token', '').strip()

    if not api_key or not hf_api_token:
        return jsonify({'error': 'Both Gemini API Key and Hugging Face Token are required.'}), 400

    try:
        # Create and store a genai.Client that we reuse for requests.
        # If you want Vertex AI instead of the Gemini Developer API, use:
        # genai.Client(vertexai=True, api_key=api_key, project="...", location="us-central1")
        genai_client = genai.Client(api_key=api_key)

        # Optionally you can validate by making a lightweight call, but avoid doing heavy calls here.
        # test = genai_client.models.generate_content(model=DEFAULT_GEMINI_MODEL, contents="Hello")
        return jsonify({'message': 'API keys set successfully.'}), 200
    except Exception as e:
        return jsonify({'error': f'Failed to create GenAI client: {str(e)}'}), 400


@app.route('/api/chat', methods=['POST'])
def chat():
    global api_key, hf_api_token, genai_client

    if not api_key or not hf_api_token or genai_client is None:
        return jsonify({'error': 'API keys not set. Please set the API keys first.'}), 400

    data = request.json
    input_prompt = data.get('prompt', '').strip()
    history_data = data.get('history', [])

    if not input_prompt:
        return jsonify({'error': 'Please enter a valid prompt.'}), 400

    try:
        # IMAGE generation path (unchanged)
        if input_prompt.lower().startswith("generate image of"):
            image_prompt = input_prompt[len("generate image of"):].strip()

            if not hf_api_token:
                return jsonify({'error': 'Hugging Face Token not set.'}), 400

            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {hf_api_token}"}
            payload = {"inputs": image_prompt}

            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            image_bytes = response.content
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            image_data_url = f"data:image/jpeg;base64,{base64_image}"

            return jsonify({'image_url': image_data_url}), 200

        # TEXT generation path using google-genai SDK
        # Build structured contents list using types.Content / types.Part
        contents = []

        # Role mapping: accept 'user', 'model' (from your frontend), 'assistant', 'system'
        role_map = {
            'user': 'user',
            'model': 'assistant',
            'assistant': 'assistant',
            'system': 'system'
        }

        # Convert history entries to types.Content objects
        for entry in history_data:
            # Expecting history entries of the shape { type: 'user'|'model'|'assistant'|'system', text: '...' }
            incoming_role = entry.get('type', 'user')
            role = role_map.get(incoming_role, 'user')
            text = entry.get('text', '')
            if not text:
                continue
            part = types.Part(text=text)
            content = types.Content(role=role, parts=[part])
            contents.append(content)

        # Add current user prompt
        user_part = types.Part(text=input_prompt)
        user_content = types.Content(role='user', parts=[user_part])
        contents.append(user_content)

        # Choose model name (adjust if you have a different model)
        model_name = DEFAULT_GEMINI_MODEL

        # Make the call with the reusable client
        response = genai_client.models.generate_content(
            model=model_name,
            contents=contents
        )

        # Extract text safely: prefer response.text (convenience), fallback to iterating output parts
        text_result = getattr(response, 'text', None)
        if not text_result:
            # Try to build text from response.output -> content -> parts -> text (structure may vary)
            try:
                parts_accum = []
                for out_item in getattr(response, 'output', []) or []:
                    # out_item might be a Content-like object with a 'parts' attribute
                    for part in getattr(out_item, 'parts', []) or []:
                        ptext = getattr(part, 'text', None)
                        if ptext:
                            parts_accum.append(ptext)
                if parts_accum:
                    text_result = "\n".join(parts_accum)
                else:
                    # last resort: stringify response
                    text_result = str(response)
            except Exception:
                text_result = str(response)

        return jsonify({'response': text_result}), 200

    except requests.exceptions.HTTPError as e:
        # Hugging Face image errors handled here
        if e.response is not None and e.response.status_code == 503:
            return jsonify({'error': 'The image model is currently loading, please try again in 20-30 seconds.'}), 503
        elif e.response is not None and e.response.status_code == 401:
            return jsonify({'error': 'Authentication failed. Please check your Hugging Face token.'}), 401
        return jsonify({'error': f"Error calling Image API: {str(e)}"}), 500
    except Exception as e:
        # GenAI or other runtime errors
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
