from groq import Groq
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import base64
import mimetypes
import requests
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

load_dotenv()

def get_uri(image_url: str):
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    image_encoded = base64.b64encode(image_bytes).decode('utf-8')
    image_ext, _ = mimetypes.guess_type(image_url)
    data_uri = f"data:{image_ext};base64,{image_encoded}" 
    return data_uri

class State(TypedDict):
    subject_url: str
    clothes_url: str
    subject_description: str
    garment_description: str
    subject_clothes_type: str
    clothes_type: str
    upper_clothes: bool
    pants: bool
    skirt: bool
    dress: bool
    left_arm: bool
    right_arm: bool
    left_leg: bool
    right_leg: bool
    lower_neck: bool
    errors: Annotated[list, operator.add]

class VLMAgent:
    def __init__(self, client: Groq):
        self.client = client
    
    def describe_image(self, image_uri: str, image_type: str) -> str:
        if image_type == "subject image":
            prompt = """Describe the person's current outfit in detail:

1. UPPER GARMENT:
   - Exact type (t-shirt, blouse, hoodie, blazer, tank top, crop top, dress, jumpsuit, etc.)
   - If layered, identify ALL layers (e.g., "blazer over white t-shirt")
   - Sleeve type (sleeveless/tank, short-sleeve, 3/4 sleeve, long-sleeve, off-shoulder)
   - Neckline (crew neck, v-neck, off-shoulder, high neck, etc.)

2. LOWER GARMENT:
   - Exact type (shorts, pants, jeans, skirt, dress, jumpsuit, underwear, etc.)
   - Length (mini/short, knee-length, mid-calf/capri, full-length/maxi)
   - If wearing a one-piece (dress/jumpsuit), clearly state "one-piece dress" or "jumpsuit"

3. STRUCTURE:
   - Is this separates (top + bottom) or one-piece (dress/jumpsuit)?
   - Clearly state: "wearing separates" or "wearing one-piece dress" or "wearing jumpsuit"

Format: "[upper garment details], [lower garment details], [structure]"
Example: "rust-colored long-sleeve hoodie, black full-length pants, wearing separates"
Example: "white off-shoulder short dress, wearing one-piece dress"
Example: "light green long-sleeve blazer over white t-shirt, dark blue full-length jeans, wearing separates"""
        else:
            prompt = """Describe this new garment in detail:

1. GARMENT TYPE:
   - Exact category (t-shirt, hoodie, shirt, blouse, pants, shorts, skirt, dress, jumpsuit, bra, etc.)
   - Is it a TOP only, BOTTOM only, or ONE-PIECE (dress/jumpsuit)?
   - Clearly state the category

2. SLEEVE/ARM COVERAGE (if applicable):
   - Sleeve type (sleeveless/tank, short-sleeve, 3/4 sleeve, long-sleeve, off-shoulder)
   - Neckline type

3. LEG COVERAGE (if applicable):
   - Length (short/mini, knee-length, mid-calf/capri, full-length/maxi)

Format: "[garment type], [sleeve details if top], [length details if bottom/dress]"
Example: "light blue long-sleeve hoodie, long sleeves ending at wrist"
Example: "dark blue full-length pants, full-length ending at ankle"
Example: "pink full-length jumpsuit with long sleeves, long sleeves ending at wrist, full-length ending at ankle"
Example: "red short-sleeve soccer jersey, short sleeves ending at mid-upper-arm"
Example: "maxi skirt, full-length ending at ankle"
Example: "black bra and underwear, sleeveless top, minimal leg coverage"""

        completion = self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_uri}}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=500
        )
        return completion.choices[0].message.content

class GarmentTypeAgent:
    def __init__(self, client: Groq):
        self.client = client
    
    def analyze(self, subject_desc: str, garment_desc: str) -> dict:
        prompt = f"""Analyze garment replacement for virtual try-on:

SUBJECT WEARING: {subject_desc}
NEW GARMENT: {garment_desc}

CRITICAL CONCEPT:
The booleans indicate which parts of the SUBJECT's current outfit need to be segmented/removed.
These are determined by looking at what the SUBJECT is currently wearing, then checking what parts the NEW GARMENT will replace.

STEP 1: Identify SUBJECT's current outfit structure
Ask: What is the subject wearing?
- If wearing TOP + PANTS → subject has: upper_clothes component, pants component
- If wearing TOP + SKIRT → subject has: upper_clothes component, skirt component  
- If wearing ONE-PIECE DRESS → subject has: dress component
- If wearing JUMPSUIT → subject has: upper_clothes component, pants component (jumpsuit covers both)

STEP 2: Identify what the NEW GARMENT is
Ask: What type is the new garment?
- TOP only (t-shirt, hoodie, shirt, blouse, blazer, tank, crop top, bra)
- BOTTOM only (pants, shorts, jeans)
- SKIRT only (mini, midi, maxi skirt)
- ONE-PIECE (dress or jumpsuit)

STEP 3: Determine which SUBJECT components to segment (set to true)
Based on what SUBJECT is wearing and what NEW GARMENT will replace:

GENERAL RULE FOR LIMBS:
- **If the new garment is a BOTTOM ONLY (pants/skirt), the arm arguments MUST be false.**
- **If the new garment is a TOP ONLY, the leg arguments MUST be false.**

RULE 1 - NEW GARMENT = TOP ONLY:
Look at subject: Does subject have an upper_clothes component? YES
→ Segment subject's top: upper_clothes = true
→ Keep subject's bottom: pants = false, skirt = false, dress = false

RULE 2 - NEW GARMENT = PANTS/SHORTS:
Look at subject: Does subject have a pants component? OR skirt component?
→ IF subject wears pants/shorts: Segment subject's pants → pants = true
→ IF subject wears skirt: Segment subject's skirt → skirt = true  
→ IF subject wears SEPARATES (upper_clothes + pants/shorts/skirt): Segment subject's pants → pants = true,
→ Keep subject's top and dont segment arms: upper_clothes = false, dress = false, right_arm = false, left_arm = false

RULE 3 - NEW GARMENT = SKIRT:
Look at subject: Does subject have a pants component? OR skirt component?
→ IF subject wears pants/shorts: Segment subject's pants → pants = true
→ IF subject wears skirt: Segment subject's skirt → skirt = true
→ Keep subject's top: upper_clothes = false, dress = false

RULE 4 - NEW GARMENT = DRESS:
Look at subject: What structure does subject have?
→ IF subject wears ONE-PIECE DRESS: Segment subject's dress → dress = true, others = false
→ IF subject wears TOP + PANTS: Segment both → upper_clothes = true, pants = true, skirt = false, dress = false
→ IF subject wears TOP + SKIRT: Segment both → upper_clothes = true, skirt = true, pants = false, dress = false

RULE 5 - NEW GARMENT = JUMPSUIT:
Look at subject: What structure does subject have?
→ IF subject wears ONE-PIECE DRESS: Segment subject's dress → dress = true, others = false
→ IF subject wears SEPARATES (any top + bottom): Segment both → upper_clothes = true, pants = true, skirt = false, dress = false

EXAMPLES (showing SUBJECT structure → segmentation decision):
- Subject has: "t-shirt, pants" + New: "hoodie" → Segment subject's t-shirt → upper_clothes=true
- Subject has: "hoodie, pants" + New: "shirt and pants" → Segment subject's hoodie AND pants → upper_clothes=true, pants=true
- Subject has: "blazer+t-shirt, jeans" + New: "jersey" → Segment subject's top layers → upper_clothes=true
- Subject has: "one-piece dress" + New: "jumpsuit" → Segment subject's dress → dress=true
- Subject has: "t-shirt, shorts" + New: "pants" → Segment subject's shorts → pants=true
- Subject has: "crop top, pants" + New: "bra and underwear" → Segment subject's top AND pants → upper_clothes=true, pants=true
- Subject has: "hoodie, pants" + New: "hoodie and pants" → Segment subject's hoodie AND pants → upper_clothes=true, pants=true
- Subject has: "tank, skirt" + New: "maxi dress" → Segment subject's tank AND skirt → upper_clothes=true, skirt=true
- Subject has: "blouse, skirt" + New: "skirt" → Segment subject's skirt → skirt=true
- Subject has: "tank, jeans" + New: "t-shirt" → Segment subject's tank → upper_clothes=true

Return JSON with this exact structure:
{{
    "subject_clothes_type": "copy exact subject description",
    "clothes_type": "copy exact new garment description",
    "upper_clothes": true/false,
    "pants": true/false,
    "skirt": true/false,
    "dress": true/false
}}"""

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)

class ArmSegmentAgent:
    def __init__(self, client: Groq):
        self.client = client
    
    def analyze(self, subject_desc: str, garment_desc: str) -> dict:
        prompt = f"""Analyze arm exposure for virtual try-on:

SUBJECT: {subject_desc}
NEW GARMENT: {garment_desc}

Determine left_arm and right_arm boolean values using this logic:

**OVERRIDE RULE: If the NEW GARMENT is a bottom-only piece (e.g., pants, shorts, skirt), then left_arm and right_arm MUST be false. The CORE RULE below only applies if the new garment covers the torso.**

CORE RULE: Set to TRUE only if:
1. New garment exposes LESS arm skin than subject (covers MORE)
2. AND sleeve types are NOT similar

Set to FALSE if:
- The OVERRIDE RULE applies.
- New garment exposes SAME or MORE arm skin.
- Sleeve types are similar (both short-sleeve, both long-sleeve, etc.).

Reference points for exposure:
- Fingertips = 0% exposed (100% covered)
- Wrist = 15% exposed
- Mid-forearm = 35% exposed
- Elbow = 50% exposed
- Mid-upper-arm = 75% exposed
- Shoulder = 100% exposed (sleeveless)

Think step by step:
1. First, check the OVERRIDE RULE. Is the new garment a bottom? If yes, set arms to false and stop.
2. If not, identify where subject's sleeve ends and % arm exposed.
3. Identify where new garment's sleeve ends and % arm exposed.
4. Compare: Is new garment % LESS than subject %?
5. Check: Are sleeve types similar?
6. Decide: TRUE only if new exposes LESS AND types NOT similar.

Return JSON:
{{
    "subject_arm_exposure_percent": number,
    "garment_arm_exposure_percent": number,
    "sleeve_types_similar": true/false,
    "reasoning": "brief explanation, mentioning the override rule if used",
    "left_arm": true/false,
    "right_arm": true/false
}}"""

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)

class LegSegmentAgent:
    def __init__(self, client: Groq):
        self.client = client
    
    def analyze(self, subject_desc: str, garment_desc: str) -> dict:
        prompt = f"""Analyze leg exposure for virtual try-on:

SUBJECT: {subject_desc}
NEW GARMENT: {garment_desc}

Determine left_leg and right_leg boolean values using this logic:

**OVERRIDE RULE: If the NEW GARMENT is a top-only piece (e.g., t-shirt, hoodie, blouse), then left_leg and right_leg MUST be false. The CORE RULE below only applies if the new garment is a bottom or one-piece.**

CORE RULE: Set to TRUE only if:
1. New garment exposes LESS leg skin than subject (covers MORE)
2. AND lower garment types are NOT similar

Set to FALSE if:
- The OVERRIDE RULE applies.
- New garment exposes SAME or MORE leg skin.
- Lower garment types are similar (both shorts, both full-length pants, etc.).

Reference points for exposure:
- Toes = 0% exposed (100% covered)
- Ankle = 5% exposed
- Mid-calf = 25% exposed
- Knee = 50% exposed
- Mid-thigh = 70% exposed
- Upper-thigh = 90% exposed
- Hip = 100% exposed

Think step by step:
1. First, check the OVERRIDE RULE. Is the new garment a top? If yes, set legs to false and stop.
2. If not, identify where subject's lower garment ends and % leg exposed.
3. Identify where new garment ends and % leg exposed.
4. Compare: Is new garment % LESS than subject %?
5. Check: Are lower garment types similar?
6. Decide: TRUE only if new exposes LESS AND types NOT similar.

Return JSON:
{{
    "subject_leg_exposure_percent": number,
    "garment_leg_exposure_percent": number,
    "garment_types_similar": true/false,
    "reasoning": "brief explanation, mentioning the override rule if used",
    "left_leg": true/false,
    "right_leg": true/false
}}"""

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)

class NeckSegmentAgent:
    def __init__(self, client: Groq):
        self.client = client
    
    def analyze(self, subject_desc: str, garment_desc: str) -> dict:
        prompt = f"""Analyze neckline exposure:

SUBJECT: {subject_desc}
NEW GARMENT: {garment_desc}

Determine lower_neck boolean:
- TRUE if subject's neckline is MORE exposed than new garment requires
- FALSE otherwise

Return JSON:
{{
    "reasoning": "brief explanation",
    "lower_neck": true/false
}}"""

        completion = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)

def describe_subject(state: State, client: Groq) -> State:
    vlm = VLMAgent(client)
    subject_uri = get_uri(state["subject_url"])
    state["subject_description"] = vlm.describe_image(subject_uri, "subject image")
    return state

def describe_garment(state: State, client: Groq) -> State:
    vlm = VLMAgent(client)
    garment_uri = get_uri(state["clothes_url"])
    state["garment_description"] = vlm.describe_image(garment_uri, "garment image")
    return state

def analyze_garment_type(state: State, client: Groq) -> State:
    agent = GarmentTypeAgent(client)
    result = agent.analyze(state["subject_description"], state["garment_description"])
    state["subject_clothes_type"] = result["subject_clothes_type"]
    state["clothes_type"] = result["clothes_type"]
    state["upper_clothes"] = result["upper_clothes"]
    state["pants"] = result["pants"]
    state["skirt"] = result["skirt"]
    state["dress"] = result["dress"]
    return state

def analyze_arms(state: State, client: Groq) -> State:
    agent = ArmSegmentAgent(client)
    result = agent.analyze(state["subject_description"], state["garment_description"])
    state["left_arm"] = result["left_arm"]
    state["right_arm"] = result["right_arm"]
    return state

def analyze_legs(state: State, client: Groq) -> State:
    agent = LegSegmentAgent(client)
    result = agent.analyze(state["subject_description"], state["garment_description"])
    state["left_leg"] = result["left_leg"]
    state["right_leg"] = result["right_leg"]
    return state

def analyze_neck(state: State, client: Groq) -> State:
    agent = NeckSegmentAgent(client)
    result = agent.analyze(state["subject_description"], state["garment_description"])
    state["lower_neck"] = result["lower_neck"]
    return state

def build_graph(client: Groq):
    workflow = StateGraph(State)
    
    workflow.add_node("describe_subject", lambda s: describe_subject(s, client))
    workflow.add_node("describe_garment", lambda s: describe_garment(s, client))
    workflow.add_node("analyze_garment_type", lambda s: analyze_garment_type(s, client))
    workflow.add_node("analyze_arms", lambda s: analyze_arms(s, client))
    workflow.add_node("analyze_legs", lambda s: analyze_legs(s, client))
    workflow.add_node("analyze_neck", lambda s: analyze_neck(s, client))
    
    workflow.set_entry_point("describe_subject")
    workflow.add_edge("describe_subject", "describe_garment")
    workflow.add_edge("describe_garment", "analyze_garment_type")
    workflow.add_edge("analyze_garment_type", "analyze_arms")
    workflow.add_edge("analyze_arms", "analyze_legs")
    workflow.add_edge("analyze_legs", "analyze_neck")
    workflow.add_edge("analyze_neck", END)
    
    return workflow.compile()

def get_segments(subject_url: str, clothes_url: str):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    graph = build_graph(client)
    
    initial_state = {
        "subject_url": subject_url,
        "clothes_url": clothes_url,
        "subject_description": "",
        "garment_description": "",
        "subject_clothes_type": "",
        "clothes_type": "",
        "upper_clothes": False,
        "pants": False,
        "skirt": False,
        "dress": False,
        "left_arm": False,
        "right_arm": False,
        "left_leg": False,
        "right_leg": False,
        "lower_neck": False,
        "errors": []
    }
    
    result = graph.invoke(initial_state)
    
    return {
        "subject_clothes_type": result["subject_clothes_type"],
        "clothes_type": result["clothes_type"],
        "upper_clothes": result["upper_clothes"],
        "pants": result["pants"],
        "skirt": result["skirt"],
        "dress": result["dress"],
        "left_arm": result["left_arm"],
        "right_arm": result["right_arm"],
        "left_leg": result["left_leg"],
        "right_leg": result["right_leg"],
        "lower_neck": result["lower_neck"]
    }

if __name__ == "__main__":
    subject_url = 'https://res.cloudinary.com/dukgi26uv/image/upload/v1761044493/images_3_cpj0vc.jpg'
    clothes_url = 'https://res.cloudinary.com/dukgi26uv/image/upload/v1761044493/51-M1KDRJsL._AC_UY1100__hyr2fb.jpg'
    print(json.dumps(get_segments(subject_url, clothes_url), indent=2))
