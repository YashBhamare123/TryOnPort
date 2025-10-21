from crewai import Agent, Task, Crew, Process
from langchain_anthropic import ChatAnthropic
from groq import Groq
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import base64
import mimetypes
import requests
from typing import Dict, Any


llm = ChatAnthropic(
    model_name="claude-3-haiku-20240307",
    temperature=0.0 ,
    api_key = os.environ["ANTHROPIC_API_KEY"] ,
    max_tokens=512
)


def get_uri(image_url: str):
    """Convert image URL to data URI for API consumption"""
    response = requests.get(image_url)
    response.raise_for_status()
    image_bytes = response.content
    image_encoded = base64.b64encode(image_bytes).decode('utf-8')
    
    image_ext, _ = mimetypes.guess_type(image_url)
    data_uri = f"data:{image_ext};base64,{image_encoded}" 
    return data_uri

class SegmentChoices(BaseModel):
    """Schema for segmentation output"""
    subject_clothes_type: str
    clothes_type: str
    left_arm: bool
    right_arm: bool
    left_leg: bool
    right_leg: bool
    upper_clothes: bool
    skirt: bool
    pants: bool
    dress: bool
    lower_neck: bool

class VLMInterface:
    """Interface for Vision Language Model calls"""
    
    def __init__(self):
        load_dotenv()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    def analyze_images(self, subject_uri: str, clothes_uri: str, focus_area: str) -> str:
        """
        Call VLM with specific focus instructions
        
        Args:
            subject_uri: Data URI of subject image
            clothes_uri: Data URI of garment image
            focus_area: Specific analysis focus (e.g., "arm exposure", "leg coverage")
        """
        focus_prompts = {
            "garment_identification": """
                Analyze both images and provide detailed descriptions:
                
                1. SUBJECT IMAGE - What is the person wearing?
                   - Describe the TOP garment (if wearing separates)
                   - Describe the BOTTOM garment (pants, shorts, skirt, etc.)
                   - OR describe the DRESS (if wearing one-piece)
                   - Include details: sleeve length, garment length, style
                
                2. NEW GARMENT IMAGE - What garment is being tried on?
                   - Is it a TOP (shirt, blouse, hoodie, etc.)?
                   - Is it a BOTTOM (pants, shorts, skirt)?
                   - Is it a DRESS (one-piece garment)?
                   - Include details: sleeve length, garment length, style
                
                Be very specific and descriptive.
            """,
            
            "upper_clothes_logic": """
                Determine if upper_clothes should be TRUE or FALSE.
                
                RULES:
                - TRUE if: New garment is a TOP that replaces subject's current top
                - TRUE if: New garment is a DRESS and subject wears SEPARATES
                - FALSE if: New garment is only pants/skirt/bottom
                - FALSE if: Both subject and new garment are dresses
                
                Analyze the garments and state clearly: upper_clothes = TRUE or FALSE
            """,
            
            "pants_logic": """
                Determine if pants should be TRUE or FALSE.
                
                RULES:
                - TRUE if: New garment replaces subject's existing pants/shorts/leggings
                - TRUE if: New garment is SKIRT and subject wears PANTS
                - TRUE if: New garment is DRESS and subject wears SEPARATES with PANTS
                - FALSE if: Subject wears skirt or dress (not pants)
                - FALSE if: New garment is only a top
                
                Analyze the garments and state clearly: pants = TRUE or FALSE
            """,
            
            "skirt_logic": """
                Determine if skirt should be TRUE or FALSE.
                
                RULES:
                - TRUE if: New garment replaces subject's existing skirt
                - TRUE if: New garment is PANTS and subject wears SKIRT
                - TRUE if: New garment is DRESS and subject wears SEPARATES with SKIRT
                - FALSE if: Subject wears pants (not a skirt)
                - FALSE if: New garment is only a top
                
                Analyze the garments and state clearly: skirt = TRUE or FALSE
            """,
            
            "dress_logic": """
                Determine if dress should be TRUE or FALSE.
                
                RULES:
                - TRUE if: Subject wears ONE-PIECE DRESS and new garment is SEPARATES
                - TRUE if: Subject wears DRESS and new garment replaces it with another outfit type
                - FALSE if: Subject wears separates (top + bottom)
                - FALSE if: Both subject and new garment are dresses
                
                Analyze the garments and state clearly: dress = TRUE or FALSE
            """,
            
            "arm_exposure_measurement": """
                CRITICAL: Compare ARM SKIN EXPOSURE by measuring BOTH images.
                
                VISUAL REFERENCE POINTS (% arm exposed from fingertips):
                - Fingertips/Knuckles: 0% exposed (full coverage)
                - Wrist: 15% exposed
                - Mid-forearm: 35% exposed
                - Elbow: 50% exposed
                - Mid-upper-arm: 75% exposed
                - Shoulder: 100% exposed (sleeveless)
                
                STEP-BY-STEP ANALYSIS:
                
                1. SUBJECT'S CURRENT GARMENT:
                   - Where does the sleeve end? (wrist/elbow/shoulder?)
                   - What percentage of arm is EXPOSED?
                   - What is the sleeve type? (short-sleeve/long-sleeve/sleeveless/3-quarter?)
                
                2. NEW GARMENT:
                   - Where does the sleeve end? (wrist/elbow/shoulder?)
                   - What percentage of arm would be EXPOSED?
                   - What is the sleeve type? (short-sleeve/long-sleeve/sleeveless/3-quarter?)
                
                3. COMPARISON:
                   - Subject arm exposed: ___% 
                   - New garment arm exposed: ___%
                   - Is new < subject? (Does new expose LESS?)
                   
                4. SLEEVE TYPE CHECK:
                   - Are sleeve types similar? (e.g., both short-sleeve, both long-sleeve)
                   - If types are similar → MUST be FALSE regardless of percentages
                
                Provide exact measurements and clear reasoning.
            """,
            
            "leg_exposure_measurement": """
                CRITICAL: Compare LEG SKIN EXPOSURE by measuring BOTH images.
                
                VISUAL REFERENCE POINTS (% leg exposed from toes):
                - Toes: 0% exposed (full coverage with socks)
                - Ankle: 5% exposed (full-length pants)
                - Mid-calf: 25% exposed (capri pants)
                - Knee: 50% exposed (knee-length shorts/skirt)
                - Mid-thigh: 70% exposed (short shorts/mini skirt)
                - Upper-thigh: 90% exposed (very short)
                - Hip: 100% exposed (swimwear)
                
                STEP-BY-STEP ANALYSIS:
                
                1. SUBJECT'S CURRENT GARMENT:
                   - Where does the lower garment end? (ankle/knee/thigh?)
                   - What percentage of leg is EXPOSED?
                   - What is the garment type? (full-length pants/shorts/mini skirt/maxi skirt?)
                
                2. NEW GARMENT:
                   - Where does the lower garment end? (ankle/knee/thigh?)
                   - What percentage of leg would be EXPOSED?
                   - What is the garment type? (full-length pants/shorts/mini skirt/maxi skirt?)
                
                3. COMPARISON:
                   - Subject leg exposed: ___% 
                   - New garment leg exposed: ___%
                   - Is new < subject? (Does new expose LESS?)
                   
                4. GARMENT TYPE CHECK:
                   - Are lower garment types similar? (e.g., both shorts, both full-length pants)
                   - If types are similar → MUST be FALSE regardless of percentages
                
                Provide exact measurements and clear reasoning.
            """,
            
            "neck_exposure": """
                Compare NECKLINE EXPOSURE between subject and new garment:
                
                Analyze:
                1. Subject's current neckline style and coverage
                2. New garment's neckline style and coverage
                3. Does new garment require MORE neck coverage than subject currently has?
                
                Set lower_neck TRUE if subject's neckline is more exposed than what new garment requires.
            """
        }
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "This is my subject image"},
                    {"type": "image_url", "image_url": {"url": subject_uri}}
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""This is the garment image. 
                        
                        FOCUS YOUR ANALYSIS ON: {focus_area}
                        
                        {focus_prompts.get(focus_area, '')}
                        
                        Provide detailed analysis with specific measurements and comparisons."""
                    },
                    {"type": "image_url", "image_url": {"url": clothes_uri}}
                ]
            }
        ]
        
        completion = self.client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.3
        )
        
        return completion.choices[0].message.content

def create_orchestrator_agent(vlm: VLMInterface) -> Agent:
    """Orchestrator agent that coordinates all analysis"""
    return Agent(
        role='Segmentation Orchestrator',
        goal='Coordinate the analysis of subject and garment images to determine correct segmentation choices',
        backstory="""You are the master coordinator for virtual try-on segmentation. 
        You understand the complete workflow and ensure all specialized agents work together 
        to produce accurate segmentation decisions based on skin exposure comparisons.""",
    
        allow_delegation=True,
        llm=llm
    )

def create_upper_clothes_agent(vlm: VLMInterface) -> Agent:
    """Agent responsible for upper_clothes segmentation decision"""
    return Agent(
        role='Upper Clothes Segmentation Specialist',
        goal='Determine if upper_clothes should be TRUE by analyzing what garment types are present',
        backstory="""You are an expert in determining when a subject's top needs to be segmented out.
        
        YOUR TASK: Decide upper_clothes boolean based ONLY on garment type analysis.
        
        LOGIC:
        - TRUE if: New garment is a TOP (shirt, blouse, hoodie, etc.) that replaces subject's separate top
        - TRUE if: New garment is a DRESS and subject wears SEPARATES (top + bottom)
        - FALSE if: New garment is only a BOTTOM (pants/skirt) and doesn't affect the top
        - FALSE if: Both subject and new garment are dresses
        
        EXAMPLES:
        ✓ Subject: T-shirt + Pants | New: Hoodie → TRUE (top replaces top)
        ✓ Subject: Blouse + Skirt | New: Dress → TRUE (dress replaces separates, need to remove top)
        ✗ Subject: Shirt + Pants | New: Skirt → FALSE (only bottom changes)
        ✗ Subject: Dress | New: Dress → FALSE (dress replaces dress directly)
        """,

        allow_delegation=False,
        llm=llm
    )

def create_pants_agent(vlm: VLMInterface) -> Agent:
    """Agent responsible for pants segmentation decision"""
    return Agent(
        role='Pants Segmentation Specialist',
        goal='Determine if pants should be TRUE by analyzing garment types and what needs replacement',
        backstory="""You are an expert in determining when a subject's pants need to be segmented out.
        
        YOUR TASK: Decide pants boolean based ONLY on garment type analysis.
        
        LOGIC:
        - TRUE if: Subject wears PANTS/SHORTS and new garment replaces them (new is pants, skirt, or dress)
        - TRUE if: Subject wears PANTS and new garment is a SKIRT (replacing pants with skirt)
        - TRUE if: Subject wears PANTS and new garment is a DRESS (dress replaces separates including pants)
        - FALSE if: Subject wears SKIRT or DRESS (not pants)
        - FALSE if: New garment is only a TOP (top doesn't affect pants)
        
        EXAMPLES:
        ✓ Subject: Shirt + Pants | New: Full-length pants → TRUE (pants replace pants)
        ✓ Subject: Top + Shorts | New: Skirt → TRUE (skirt replaces shorts/pants)
        ✓ Subject: Blouse + Pants | New: Dress → TRUE (dress replaces separates including pants)
        ✗ Subject: Shirt + Skirt | New: Top → FALSE (subject wears skirt, not pants)
        ✗ Subject: Dress | New: Top → FALSE (subject wears dress, not pants)
        """,
        allow_delegation=False,
        llm=llm
    )

def create_skirt_agent(vlm: VLMInterface) -> Agent:
    """Agent responsible for skirt segmentation decision"""
    return Agent(
        role='Skirt Segmentation Specialist',
        goal='Determine if skirt should be TRUE by analyzing garment types and what needs replacement',
        backstory="""You are an expert in determining when a subject's skirt needs to be segmented out.
        
        YOUR TASK: Decide skirt boolean based ONLY on garment type analysis.
        
        LOGIC:
        - TRUE if: Subject wears SKIRT and new garment replaces it (new is pants, skirt, or dress)
        - TRUE if: Subject wears SKIRT and new garment is PANTS (replacing skirt with pants)
        - TRUE if: Subject wears SKIRT and new garment is a DRESS (dress replaces separates including skirt)
        - FALSE if: Subject wears PANTS or DRESS (not a skirt)
        - FALSE if: New garment is only a TOP (top doesn't affect skirt)
        
        EXAMPLES:
        ✓ Subject: Blouse + Mini skirt | New: Maxi skirt → TRUE (skirt replaces skirt)
        ✓ Subject: Top + Skirt | New: Pants → TRUE (pants replace skirt)
        ✓ Subject: Shirt + Skirt | New: Dress → TRUE (dress replaces separates including skirt)
        ✗ Subject: Shirt + Pants | New: Top → FALSE (subject wears pants, not skirt)
        ✗ Subject: Dress | New: Top → FALSE (subject wears dress, not skirt)
        """,
        allow_delegation=False,
        llm=llm
    )

def create_dress_agent(vlm: VLMInterface) -> Agent:
    """Agent responsible for dress segmentation decision"""
    return Agent(
        role='Dress Segmentation Specialist',
        goal='Determine if dress should be TRUE by analyzing garment types and what needs replacement',
        backstory="""You are an expert in determining when a subject's dress needs to be segmented out.
        
        YOUR TASK: Decide dress boolean based ONLY on garment type analysis.
        
        LOGIC:
        - TRUE if: Subject wears ONE-PIECE DRESS and new garment is SEPARATES (top and/or bottom)
        - TRUE if: Subject wears DRESS and new garment replaces it with separate pieces
        - FALSE if: Subject wears SEPARATES (top + bottom, not a dress)
        - FALSE if: Both subject and new garment are dresses (dress replaces dress directly)
        
        EXAMPLES:
        ✓ Subject: Knee-length dress | New: Shirt + Pants → TRUE (separates replace dress)
        ✓ Subject: Maxi dress | New: Top → TRUE (even just top means dress needs removal)
        ✗ Subject: T-shirt + Skirt | New: Dress → FALSE (subject wears separates, not dress)
        ✗ Subject: Dress | New: Dress → FALSE (dress replaces dress directly)
        """,
        allow_delegation=False,
        llm=llm
    )

def create_arm_analysis_agent(vlm: VLMInterface) -> Agent:
    """Agent responsible for arm exposure analysis - BOTH left_arm AND right_arm"""
    return Agent(
        role='Arm Exposure Analyst',
        goal='Determine if BOTH left_arm AND right_arm should be TRUE by comparing exposed arm skin between subject and new garment',
        backstory="""You are a specialist in analyzing arm coverage through precise visual measurement.
        
        YOUR TASK: Decide left_arm and right_arm booleans based ONLY on image comparison.
        
        CRITICAL RULES (BOTH CONDITIONS MUST BE MET):
        1. New garment must expose LESS arm skin than subject (new % < subject %)
        2. Sleeve types must NOT be similar (e.g., short-sleeve vs long-sleeve is different; both short-sleeve is similar)
        
        MEASUREMENT REFERENCE POINTS:
        - Fingertips: 0% exposed (full sleeve)
        - Wrist: 15% exposed (long sleeve)
        - Mid-forearm: 35% exposed
        - Elbow: 50% exposed (short sleeve)
        - Mid-upper-arm: 75% exposed
        - Shoulder: 100% exposed (sleeveless)
        
        DECISION LOGIC:
        - If new exposes LESS (new % < subject %) AND sleeve types NOT similar → left_arm = TRUE, right_arm = TRUE
        - If new exposes SAME or MORE (new % >= subject %) → left_arm = FALSE, right_arm = FALSE
        - If sleeve types ARE similar (both short-sleeve, both long-sleeve, etc.) → left_arm = FALSE, right_arm = FALSE
        
        EXAMPLES:
        ✓ Subject: Short-sleeve (50% exposed) + New: Long-sleeve (15% exposed) → TRUE (different types, less exposure)
        ✗ Subject: Long-sleeve (15% exposed) + New: Long-sleeve (15% exposed) → FALSE (similar types)
        ✗ Subject: Long-sleeve (15% exposed) + New: Sleeveless (100% exposed) → FALSE (more exposure)
        
        IMPORTANT: left_arm and right_arm ALWAYS have the SAME value. You analyze arm exposure as a whole.
        """,
        allow_delegation=False,
        llm=llm
    )

def create_leg_analysis_agent(vlm: VLMInterface) -> Agent:
    """Agent responsible for leg exposure analysis - BOTH left_leg AND right_leg"""
    return Agent(
        role='Leg Exposure Analyst',
        goal='Determine if BOTH left_leg AND right_leg should be TRUE by comparing exposed leg skin between subject and new garment',
        backstory="""You are a specialist in analyzing leg coverage through precise visual measurement.
        
        YOUR TASK: Decide left_leg and right_leg booleans based ONLY on image comparison.
        
        CRITICAL RULES (BOTH CONDITIONS MUST BE MET):
        1. New garment must expose LESS leg skin than subject (new % < subject %)
        2. Lower garment types must NOT be similar (e.g., shorts vs full-length pants is different; both full-length pants is similar)
        
        MEASUREMENT REFERENCE POINTS:
        - Toes: 0% exposed (full coverage)
        - Ankle: 5% exposed (full-length pants/maxi skirt)
        - Mid-calf: 25% exposed (capri pants/midi skirt)
        - Knee: 50% exposed (knee-length shorts/skirt)
        - Mid-thigh: 70% exposed (short shorts/mini skirt)
        - Upper-thigh: 90% exposed (very short)
        - Hip: 100% exposed (swimwear)
        
        DECISION LOGIC:
        - If new exposes LESS (new % < subject %) AND garment types NOT similar → left_leg = TRUE, right_leg = TRUE
        - If new exposes SAME or MORE (new % >= subject %) → left_leg = FALSE, right_leg = FALSE
        - If garment types ARE similar (both shorts, both full-length, etc.) → left_leg = FALSE, right_leg = FALSE
        
        EXAMPLES:
        ✓ Subject: Shorts (50% exposed) + New: Full-length pants (5% exposed) → TRUE (different types, less exposure)
        ✗ Subject: Full-length pants (5% exposed) + New: Full-length pants (5% exposed) → FALSE (similar types)
        ✗ Subject: Full-length pants (5% exposed) + New: Mini skirt (70% exposed) → FALSE (more exposure)
        
        IMPORTANT: left_leg and right_leg ALWAYS have the SAME value. You analyze leg exposure as a whole.
        """,
        allow_delegation=False,
        llm=llm
    )

def create_neck_analysis_agent(vlm: VLMInterface) -> Agent:
    """Agent responsible for neckline analysis"""
    return Agent(
        role='Neckline Coverage Analyst',
        goal='Determine if lower neck segmentation is needed based on neckline comparison',
        backstory="""You are a specialist in analyzing neckline styles and coverage. 
        You determine if the subject's current neckline is more exposed than what the 
        new garment requires, which would necessitate lower neck segmentation.""",
        allow_delegation=False,
        llm=llm
    )

def create_synthesis_agent() -> Agent:
    """Agent responsible for final synthesis"""
    return Agent(
        role='Decision Synthesis Specialist',
        goal='Combine all agent analyses into final segmentation decision following exact schema',
        backstory="""You are the final decision maker who synthesizes all specialized 
        analyses into a coherent segmentation plan. You ensure all decisions follow 
        the core logic: segments are TRUE only when new garment covers MORE (exposes LESS) 
        than subject's current garment AND garment types are not similar. You produce 
        the final JSON output in the exact required schema.""",
        allow_delegation=False,
        llm=llm
    )

def get_segments_with_agents(subject_url: str, clothes_url: str) -> Dict[str, Any]:
    """
    Main function using CrewAI agentic system for segmentation analysis
    
    Args:
        subject_url: URL of subject image
        clothes_url: URL of garment image
        
    Returns:
        Dictionary with segmentation choices following SegmentChoices schema
    """
    # Initialize VLM interface
    vlm = VLMInterface()
    
    # Convert images to data URIs
    subject_uri = get_uri(subject_url)
    clothes_uri = get_uri(clothes_url)
    
    # Create agents
    orchestrator = create_orchestrator_agent(vlm)
    arm_analyst = create_arm_analysis_agent(vlm)
    leg_analyst = create_leg_analysis_agent(vlm)
    neck_analyst = create_neck_analysis_agent(vlm)
    upper_clothes_specialist = create_upper_clothes_agent(vlm)
    pants_specialist = create_pants_agent(vlm)
    skirt_specialist = create_skirt_agent(vlm)
    dress_specialist = create_dress_agent(vlm)
    synthesizer = create_synthesis_agent()
    
    # Get garment descriptions first
    garment_analysis = vlm.analyze_images(subject_uri, clothes_uri, 'garment_identification')
    
    # Create tasks
    identify_task = Task(
        description=f"""Identify and describe the garments in both images.
        
        VLM Analysis: {garment_analysis}
        
        Provide:
        - subject_clothes_type: Complete description of what subject is wearing
        - clothes_type: Complete description of the new garment
        """,
        expected_output="Detailed descriptions: subject_clothes_type and clothes_type",
        agent=orchestrator
    )
    
    upper_clothes_task = Task(
        description=f"""Determine if upper_clothes should be TRUE or FALSE.
        
        Garment Context: {garment_analysis}
        
        YOUR DECISION PROCESS:
        
        1. Identify garment types:
           - What is subject wearing on top? (separate top / part of dress)
           - What is the new garment? (top / bottom / dress)
        
        2. Apply logic:
           - Is new garment a TOP replacing subject's separate top? → TRUE
           - Is new garment a DRESS and subject wears SEPARATES? → TRUE
           - Is new garment only a BOTTOM? → FALSE
           - Are both subject and new garment dresses? → FALSE
        
        Output ONLY:
        upper_clothes: <true/false>
        Reasoning: <brief explanation>
        """,
        expected_output="upper_clothes: true or false with reasoning",
        agent=upper_clothes_specialist
    )
    
    pants_task = Task(
        description=f"""Determine if pants should be TRUE or FALSE.
        
        Garment Context: {garment_analysis}
        
        YOUR DECISION PROCESS:
        
        1. Identify garment types:
           - What is subject wearing on bottom? (pants/shorts / skirt / part of dress)
           - What is the new garment? (top / pants/shorts / skirt / dress)
        
        2. Apply logic:
           - Does subject wear PANTS/SHORTS and new replaces them? → TRUE
           - Does subject wear PANTS and new is SKIRT? → TRUE
           - Does subject wear PANTS and new is DRESS? → TRUE
           - Does subject wear SKIRT or DRESS? → FALSE
           - Is new only a TOP? → FALSE
        
        Output ONLY:
        pants: <true/false>
        Reasoning: <brief explanation>
        """,
        expected_output="pants: true or false with reasoning",
        agent=pants_specialist
    )
    
    skirt_task = Task(
        description=f"""Determine if skirt should be TRUE or FALSE.
        
        Garment Context: {garment_analysis}
        
        YOUR DECISION PROCESS:
        
        1. Identify garment types:
           - What is subject wearing on bottom? (pants/shorts / skirt / part of dress)
           - What is the new garment? (top / pants/shorts / skirt / dress)
        
        2. Apply logic:
           - Does subject wear SKIRT and new replaces it? → TRUE
           - Does subject wear SKIRT and new is PANTS? → TRUE
           - Does subject wear SKIRT and new is DRESS? → TRUE
           - Does subject wear PANTS or DRESS? → FALSE
           - Is new only a TOP? → FALSE
        
        Output ONLY:
        skirt: <true/false>
        Reasoning: <brief explanation>
        """,
        expected_output="skirt: true or false with reasoning",
        agent=skirt_specialist
    )
    
    dress_task = Task(
        description=f"""Determine if dress should be TRUE or FALSE.
        
        Garment Context: {garment_analysis}
        
        YOUR DECISION PROCESS:
        
        1. Identify garment types:
           - Is subject wearing a ONE-PIECE DRESS or SEPARATES?
           - Is the new garment a DRESS or SEPARATES?
        
        2. Apply logic:
           - Does subject wear DRESS and new is SEPARATES (top/bottom)? → TRUE
           - Does subject wear DRESS and new replaces with separate pieces? → TRUE
           - Does subject wear SEPARATES? → FALSE
           - Are both subject and new garment dresses? → FALSE
        
        Output ONLY:
        dress: <true/false>
        Reasoning: <brief explanation>
        """,
        expected_output="dress: true or false with reasoning",
        agent=dress_specialist
    )
    
    arm_task = Task(
        description=f"""Analyze ARM EXPOSURE by comparing subject and new garment images.
        
        VLM MEASUREMENT ANALYSIS:
        {vlm.analyze_images(subject_uri, clothes_uri, 'arm_exposure_measurement')}
        
        YOUR DECISION PROCESS:
        
        1. Extract from VLM analysis:
           - Subject arm exposure: ___% (where does sleeve end?)
           - New garment arm exposure: ___% (where would sleeve end?)
           - Subject sleeve type: ___ (short-sleeve/long-sleeve/sleeveless/3-quarter?)
           - New sleeve type: ___ (short-sleeve/long-sleeve/sleeveless/3-quarter?)
        
        2. Check TWO conditions:
           Condition A: Is new % < subject %? (Does new expose LESS?)
           Condition B: Are sleeve types DIFFERENT? (Not both short-sleeve, not both long-sleeve, etc.)
        
        3. Make decision:
           - If BOTH conditions are TRUE → left_arm: true, right_arm: true
           - If EITHER condition is FALSE → left_arm: false, right_arm: false
        
        CRITICAL REMINDERS:
        - Similar sleeve types ALWAYS mean FALSE (even if percentages differ)
        - Same or more exposure ALWAYS means FALSE
        - left_arm and right_arm ALWAYS have the SAME value
        
        Output format (MUST include both):
        left_arm: <true/false>
        right_arm: <true/false>
        Reasoning: <brief explanation>
        """,
        expected_output="left_arm and right_arm booleans with reasoning",
        agent=arm_analyst
    )
    
    leg_task = Task(
        description=f"""Analyze LEG EXPOSURE by comparing subject and new garment images.
        
        VLM MEASUREMENT ANALYSIS:
        {vlm.analyze_images(subject_uri, clothes_uri, 'leg_exposure_measurement')}
        
        YOUR DECISION PROCESS:
        
        1. Extract from VLM analysis:
           - Subject leg exposure: ___% (where does lower garment end?)
           - New garment leg exposure: ___% (where would lower garment end?)
           - Subject garment type: ___ (shorts/full-length pants/mini skirt/maxi skirt?)
           - New garment type: ___ (shorts/full-length pants/mini skirt/maxi skirt?)
        
        2. Check TWO conditions:
           Condition A: Is new % < subject %? (Does new expose LESS?)
           Condition B: Are garment types DIFFERENT? (Not both shorts, not both full-length, etc.)
        
        3. Make decision:
           - If BOTH conditions are TRUE → left_leg: true, right_leg: true
           - If EITHER condition is FALSE → left_leg: false, right_leg: false
        
        CRITICAL REMINDERS:
        - Similar garment types ALWAYS mean FALSE (even if percentages differ)
        - Same or more exposure ALWAYS means FALSE
        - left_leg and right_leg ALWAYS have the SAME value
        
        Output format (MUST include both):
        left_leg: <true/false>
        right_leg: <true/false>
        Reasoning: <brief explanation>
        """,
        expected_output="left_leg and right_leg booleans with reasoning",
        agent=leg_analyst
    )
    
    neck_task = Task(
        description=f"""Determine if lower_neck should be TRUE or FALSE.
        
        Garment Context: {garment_analysis}
        
        YOUR DECISION PROCESS:
        
        1. Compare necklines:
           - Subject's current neckline: How exposed? (v-neck/crew neck/high neck/etc.)
           - New garment's neckline: How much coverage? (v-neck/crew neck/high neck/etc.)
        
        2. Apply logic:
           - Is subject's neckline MORE exposed than what new garment requires? → TRUE
           - Otherwise → FALSE
        
        Output ONLY:
        lower_neck: <true/false>
        Reasoning: <brief explanation>
        """,
        expected_output="lower_neck: true or false with reasoning",
        agent=neck_analyst
    )
    
    synthesis_task = Task(
        description="""Synthesize all agent analyses into final JSON output ONLY.
        
        Combine results from all specialists:
        - Orchestrator: subject_clothes_type, clothes_type descriptions
        - Upper Clothes Specialist: upper_clothes boolean
        - Pants Specialist: pants boolean
        - Skirt Specialist: skirt boolean
        - Dress Specialist: dress boolean
        - Arm Analyst: left_arm, right_arm booleans
        - Leg Analyst: left_leg, right_leg booleans
        - Neck Analyst: lower_neck boolean
        
        Output ONLY a valid JSON object, no explanations, no additional text:
        {
            "subject_clothes_type": "string",
            "clothes_type": "string",
            "upper_clothes": boolean,
            "pants": boolean,
            "skirt": boolean,
            "dress": boolean,
            "left_arm": boolean,
            "right_arm": boolean,
            "left_leg": boolean,
            "right_leg": boolean,
            "lower_neck": boolean
        }
        """,
        expected_output="Pure JSON object only, nothing else",
        agent=synthesizer
    )
    
    # Create crew
    crew = Crew(
        agents=[
            orchestrator,
            upper_clothes_specialist,
            pants_specialist, 
            skirt_specialist,
            dress_specialist,
            arm_analyst,
            leg_analyst,
            neck_analyst,
            synthesizer
        ],
        tasks=[
            identify_task,
            upper_clothes_task,
            pants_task,
            skirt_task,
            dress_task,
            arm_task,
            leg_task,
            neck_task,
            synthesis_task
        ],
        process=Process.sequential,

    )
    
    # Execute crew
    result = crew.kickoff()
    
    # Extract only the final JSON output
    result_str = str(result).strip()
    
    # Try to parse the JSON output
    try:
        # Find the last complete JSON object in the output
        start_idx = result_str.rfind('{')
        end_idx = result_str.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = result_str[start_idx:end_idx]
            segments = json.loads(json_str)
            
            # Validate schema
            SegmentChoices(**segments)
            return segments
        else:
            raise ValueError("No JSON object found in output")
            
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing crew output: {e}")
        print(f"Raw output: {result_str[:500]}...")
        
        # Return a properly structured response with all required fields
        raise Exception(f"Failed to parse valid JSON from crew output: {e}")


if __name__ == "__main__":
    subject_url = 'https://res.cloudinary.com/dukgi26uv/image/upload/v1754043339/tryon-images/pnw39seevmetdc1mjmar.jpg'
    clothes_url = 'https://res.cloudinary.com/dukgi26uv/image/upload/v1754043340/tryon-images/qeui9bhhkhkk4uh1636s.jpg'
    
    print("Starting CrewAI Segmentation Analysis...")
    print("=" * 80)
    result = get_segments_with_agents(subject_url, clothes_url)
    print("=" * 80)
    print("\nFinal Segmentation Result:")
    print(json.dumps(result, indent=2))