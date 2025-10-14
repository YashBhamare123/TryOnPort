from groq import Groq
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import base64
import mimetypes
import requests

def get_uri(image_url: str):
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    image_encoded = base64.b64encode(image_bytes).decode('utf-8')

    image_ext, _ = mimetypes.guess_type(image_url)
    data_uri = f"data:{image_ext};base64,{image_encoded}" 
    return data_uri

def get_segments(subject_url: str, clothes_url : str):
    load_dotenv()
    client = Groq(api_key = os.environ.get("GROQ_API_KEY"))

    subject_uri = get_uri(subject_url)
    clothes_uri = get_uri(clothes_url)

    with open("./modules/prompt.txt", 'r') as f:
        system_prompt = f.read()

    class SegmentChoices(BaseModel):
        left_arm : bool
        right_arm : bool
        left_leg : bool
        right_leg : bool
        upper_clothes : bool
        skirt : bool
        pants : bool
        dress : bool
        lower_neck : bool

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role" : "system",
                "content" : system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is my subject image"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": subject_uri
                        }
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "In my previous message was the subject image. This is my garment image. What all will I need to segment in my subject image that was given to you previously?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": clothes_uri
                        }
                    }
                ]
            }
        ],
        temperature=0.3,
        stop=None,
        response_format= {
            "type" : "json_schema",
            "json_schema" : {
                "name" : "segmentation_choices",
                "schema" : SegmentChoices.model_json_schema()
            }
        }
    )

    segments = json.loads(completion.choices[0].message.content)
    return segments


if __name__ == "__main__":
    subject_url = 'https://res.cloudinary.com/dukgi26uv/image/upload/v1754043339/tryon-images/pnw39seevmetdc1mjmar.jpg'
    clothes_url = 'https://res.cloudinary.com/dukgi26uv/image/upload/v1754143563/tryon-images/l4gx7uzvdz0ac6jqkrgn.webp'
    print(get_segments(subject_url , clothes_url))