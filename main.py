# main.py
import os
import uuid
import logging
import asyncio
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Literal

from fastapi import FastAPI, HTTPException, status, Path, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import AsyncGroq, GroqError

# -----------------------------------------------------------------------------
# LOGGING & CONFIGURATION
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debate-backend")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.critical("GROQ_API_KEY environment variable is missing!")
    raise RuntimeError("GROQ_API_KEY environment variable is required.")

# -----------------------------------------------------------------------------
# INITIALIZE APPLICATION & CLIENTS
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Real-Time AI Debate MVP Backend",
    description="Production-ready single-file backend for managing interactive AI debate sessions.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = AsyncGroq(api_key=GROQ_API_KEY)

# -----------------------------------------------------------------------------
# IN-MEMORY STORAGE
# -----------------------------------------------------------------------------
active_sessions: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# PYDANTIC SCHEMAS (REQUEST / RESPONSE MODELS)
# -----------------------------------------------------------------------------
class SessionCreateRequest(BaseModel):
    topic: str = Field(..., min_length=5, description="The topic chosen for the debate.", examples=["Should AI replace teachers?"])
    position: Literal["FOR", "AGAINST"] = Field(..., description="The user's stance on the debate topic.", examples=["FOR"])
    duration_minutes: int = Field(..., ge=1, le=60, description="The intended length of the debate in minutes.", examples=[5])

class SessionCreateResponse(BaseModel):
    session_id: str = Field(..., description="Unique identification token for the session.")
    ai_position: Literal["FOR", "AGAINST"] = Field(..., description="The automatically assigned stance for the AI.")
    status: Literal["ACTIVE", "ENDED"] = Field(..., description="The current live state of the session.")

class TranscriptEntry(BaseModel):
    speaker: Literal["user", "ai"] = Field(..., description="The origin source of the message utterance.")
    message: str = Field(..., description="The literal transcription content.")
    timestamp: str = Field(..., description="ISO formatted timestamp indicating statement recording.")

class OpeningStatementResponse(BaseModel):
    response: str = Field(..., description="The generated opening statement text from the AI debater.")
    transcript_entry: TranscriptEntry = Field(..., description="The formal ledger tracking object stored within history.")

class DebateMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The text statement passed up from user text-to-speech tracking.", examples=["Social media causes anxiety in teenagers."])

class DebateTurnResponse(BaseModel):
    response: str = Field(..., description="The structured contextual rebuttal statement from the AI.")
    history_length: int = Field(..., description="Total cumulative message count tracking depth for internal payload integrity.")

class TranscriptResponse(BaseModel):
    session_id: str = Field(..., description="Target session correlation string.")
    messages: List[TranscriptEntry] = Field(..., description="Sequential ordered list of all statements passed inside the channel.")

class SessionStateResponse(BaseModel):
    topic: str = Field(..., description="The specified foundational debate topic.")
    user_position: Literal["FOR", "AGAINST"] = Field(..., description="Stance of human client participant.")
    ai_position: Literal["FOR", "AGAINST"] = Field(..., description="Stance of LLM agent participant.")
    status: Literal["ACTIVE", "ENDED"] = Field(..., description="Current interaction flag setting.")
    duration_minutes: int = Field(..., description="Total time boundary constraint originally provisioned.")
    message_count: int = Field(..., description="Aggregate item count historically archived inside this instance record.")

class EndDebateResponse(BaseModel):
    status: Literal["ENDED"] = Field(..., description="Terminal state conformation parameter.")

class SessionListItem(BaseModel):
    session_id: str
    topic: str
    status: str
    created_at: str

# --- MILESTONE 2: ANALYSIS ADDITIONS ---

class DebateScore(BaseModel):
    argument_strength: int = Field(..., ge=0, le=100, description="Score evaluating depth of arguments presented.", examples=[85])
    rebuttal_quality: int = Field(..., ge=0, le=100, description="Score evaluating response counter-points.", examples=[78])
    logical_reasoning: int = Field(..., ge=0, le=100, description="Score measuring structural validity and consistency.", examples=[90])
    evidence_usage: int = Field(..., ge=0, le=100, description="Score tracking data validation or contextual assertions.", examples=[65])
    persuasiveness: int = Field(..., ge=0, le=100, description="Score indicating structural impact and rhetorical style.", examples=[80])
    clarity: int = Field(..., ge=0, le=100, description="Score assessing articulation speed and word organization.", examples=[88])
    overall_score: int = Field(..., ge=0, le=100, description="Calculated summary performance index rating.", examples=[81])

class ReportStrength(BaseModel):
    title: str = Field(..., description="High-level category label.", examples=["Effective Analogies"])
    description: str = Field(..., description="Deep specific analysis tied back to structural occurrences.", examples=["Your application of tutoring frameworks supported your opening positions cleanly."])

class ReportWeakness(BaseModel):
    title: str = Field(..., description="Identified area of logical concession.", examples=["Defensive Tone Shift"])
    description: str = Field(..., description="Detailed analytical review mapping exactly to transcript segments.", examples=["When challenged on technical parity, the response skipped analytical defenses entirely."])

class RebuttalSuggestion(BaseModel):
    original_argument: str = Field(..., description="The actual user argument processed.", examples=["AI is better because it knows everything instantly."])
    improved_rebuttal: str = Field(..., description="An optimized candidate structure recommended by the coach.", examples=["While AI possesses instant data retrieval, human teachers evaluate dynamic context structures better."])

class CoachingRecommendation(BaseModel):
    category: str = Field(..., description="Targeted focus area.", examples=["Rhetorical Framing"])
    recommendation: str = Field(..., description="Actionable training tasks.", examples=["Integrate empirical validation indicators early in turn transitions."])

class LogicalFallacy(BaseModel):
    fallacy: str = Field(..., description="The classified historical fallacy type.", examples=["Ad Hominem"])
    example: str = Field(..., description="The exact user phrase exhibiting the logical mistake.", examples=["You are just a machine so your feedback doesn't matter."])

class DebateAnalysisReport(BaseModel):
    scores: DebateScore = Field(..., description="Segmented score ledger indices.")
    strengths: List[ReportStrength] = Field(..., description="Curated item collection highlighting successful points.")
    weaknesses: List[ReportWeakness] = Field(..., description="Identified structural points requiring improvement.")
    rebuttal_suggestions: List[RebuttalSuggestion] = Field(..., description="Actionable before-and-after transcript modifications.")
    coaching_recommendations: List[CoachingRecommendation] = Field(..., description="Strategic practice suggestions.")
    logical_fallacies: List[LogicalFallacy] = Field(..., description="Exposed systemic argumentation errors.")
    summary: str = Field(..., description="Comprehensive pedagogical assessment summary.", examples=["The speaker displayed excellent structural vocabulary but structural assertions remained unbacked by contextual frameworks."])

class AnalysisStatusResponse(BaseModel):
    status: Literal["NOT_STARTED", "PROCESSING", "COMPLETED", "FAILED"] = Field(..., description="Current computation phase.")

# -----------------------------------------------------------------------------
# ROBUST GROQ LLM SERVICE HELPER
# -----------------------------------------------------------------------------
async def generate_ai_response(
    topic: str,
    ai_position: str,
    user_position: str,
    conversation_history: List[Dict[str, str]],
    latest_user_message: Optional[str] = None,
    is_opening: bool = False
) -> str:
    max_retries = 3
    retry_delay = 1.0
    timeout_duration = 15.0

    system_prompt = (
        f"You are an expert, highly persuasive, staff-level competitive backend debater.\n"
        f"Debate Topic: {topic}\n"
        f"Your Assigned Stance: {ai_position}\n"
        f"User Stance: {user_position}\n\n"
        f"CRITICAL COMPLIANCE RULES:\n"
        f"1. You MUST stay strictly on your assigned side ({ai_position}). Never switch sides or concede.\n"
        f"2. Challenge any logical flaws, unbacked claims, or assumptions made by the user directly.\n"
        f"3. Do not repeat arguments you have already used. Build dynamically off previous text.\n"
        f"4. Keep your language exceptionally conversational, natural, and fluid because your text "
        f"will be read aloud via browser voice synthesis. Avoid structured text layouts like bullet points or bold markers.\n"
    )

    if is_opening:
        system_prompt += (
            "5. This is your OPENING STATEMENT. You must explicitly establish a concise, powerful introduction "
            "framing your position. Your response MUST be strictly between 80 and 120 words. No exceptions."
        )
    else:
        system_prompt += (
            "5. This is a dynamic intermediate rebuttal turn. Respond directly, crisply, and intuitively "
            "to the user's latest claim. Be precise and keep your statement under 150 words to maintain high-speed rhythm."
        )

    messages = [{"role": "system", "content": system_prompt}]
    
    for turn in conversation_history:
        role = "assistant" if turn["speaker"] == "ai" else "user"
        messages.append({"role": role, "content": turn["message"]})

    if latest_user_message:
        messages.append({"role": "user", "content": latest_user_message})

    for attempt in range(max_retries):
        try:
            logger.info(f"Dispatching completion to Groq API. Target Stance: {ai_position}. Attempt: {attempt + 1}")
            
            response = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages, # type: ignore
                    temperature=0.73,
                    max_tokens=300,
                    top_p=0.9,
                ),
                timeout=timeout_duration
            )
            
            ai_text = response.choices[0].message.content
            if ai_text:
                return ai_text.strip()
            raise ValueError("Empty completion body received from AI provider infrastructure.")

        except (GroqError, asyncio.TimeoutError, Exception) as err:
            logger.warning(f"Groq API connection or execution event fault encountered: {str(err)}. Retrying...")
            if attempt == max_retries - 1:
                logger.error("Exhausted all available resilience retries against upstream inference cloud APIs.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Upstream debate AI generation failure occurred: {str(err)}"
                )
            await asyncio.sleep(retry_delay * (attempt + 1))
            
    return ""

# --- MILESTONE 2: CORE ANALYSIS ENGINE ---

async def generate_debate_analysis(
    topic: str,
    user_position: str,
    ai_position: str,
    transcript: List[Dict[str, Any]]
) -> DebateAnalysisReport:
    """
    Acts as an Expert Debate Coach to analyze a complete session.
    Forces JSON output generation from Groq endpoints matching the required schema.
    """
    max_retries = 3
    retry_delay = 1.5
    timeout_duration = 35.0

    # Format historical chat ledger to simple string tracking data block
    formatted_transcript_lines = []
    for entry in transcript:
        speaker_label = f"USER ({user_position})" if entry["speaker"] == "user" else f"AI ({ai_position})"
        formatted_transcript_lines.append(f"[{entry['timestamp']}] {speaker_label}: {entry['message']}")
    raw_transcript_text = "\n".join(formatted_transcript_lines)

    # NEW: Dynamically serialize the Pydantic schema to guide the LLM's structural output
    schema_json = json.dumps(DebateAnalysisReport.model_json_schema(), indent=2)

    coach_prompt = (
        "You are an Elite World-Class Debate Coach and Pedagogical Analyst.\n"
        "Your task is to comprehensively analyze the provided debate transcript and output a highly actionable coaching report.\n"
        "The report must target the USER'S performance.\n\n"
        f"Debate Context:\n"
        f"- Foundational Topic: {topic}\n"
        f"- User Assigned Stance: {user_position}\n"
        f"- Opponent AI Stance: {ai_position}\n\n"
        "CRITICAL COACHING PROTOCOLS:\n"
        "1. Evaluate precisely: Argument strength, rebuttal execution, reasoning, evidence utilization, persuasiveness, and linguistic clarity.\n"
        "2. Extract specific strengths, weaknesses, and concrete remediation recommendations. Every observation must cite context from the transcript.\n"
        "3. Explicitly detect any logical fallacies committed by the user.\n"
        "4. Provide granular before-and-after rebuttal rephrasing suggestions for weak phrases or missed arguments.\n"
        "5. You MUST return a strictly conforming JSON object matching the JSON schema blueprint below exactly. Do not output conversational prose or code blocks. All scores must be integers between 0 and 100.\n\n"
        f"REQUIRED EXPECTED JSON SCHEMA:\n{schema_json}"
    )

    messages = [
        {"role": "system", "content": coach_prompt},
        {"role": "user", "content": f"Complete Debate Chat Log:\n\"\"\"\n{raw_transcript_text}\n\"\"\""}
    ]

    for attempt in range(max_retries):
        try:
            logger.info(f"Dispatching analysis request to Groq API via structured JSON mode. Attempt: {attempt + 1}")
            
            response = await asyncio.wait_for(
                groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages, # type: ignore
                    temperature=0.2, # Low temperature to enforce strict schema adherence
                    max_tokens=2500,
                    response_format={"type": "json_object"}
                ),
                timeout=timeout_duration
            )

            response_content = response.choices[0].message.content
            if not response_content:
                raise ValueError("Received completely empty reply body from Groq analysis engine.")

            # Parse and validate the response structure against our Pydantic model
            report_data = DebateAnalysisReport.model_validate_json(response_content.strip())
            return report_data

        except (json.JSONDecodeError, ValueError, Exception) as err:
            logger.warning(f"Analysis engine parsing fault or timeout on attempt {attempt + 1}: {str(err)}")
            if attempt == max_retries - 1:
                logger.error("Analysis engine exhausted all error resilience loops.")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Analysis pipeline processing failure: {str(err)}"
                )
            await asyncio.sleep(retry_delay * (attempt + 1))

    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unreachable state in analysis generation loop.")
async def run_background_analysis_task(session_id: str):
    """
    Decoupled non-blocking worker pipeline handling analysis tasks out-of-band.
    """
    session = active_sessions.get(session_id)
    if not session:
        logger.error(f"Background worker failed to resolve tracking reference context for ID: {session_id}")
        return

    try:
        logger.info(f"Asynchronous out-of-band processing worker initialized for: {session_id}")
        report = await generate_debate_analysis(
            topic=session["topic"],
            user_position=session["user_position"],
            ai_position=session["ai_position"],
            transcript=session["conversation_history"]
        )
        session["analysis_report"] = report
        session["analysis_status"] = "COMPLETED"
        logger.info(f"Out-of-band background analysis completed successfully for ID: {session_id}")
    except Exception as e:
        session["analysis_status"] = "FAILED"
        logger.error(f"Background worker task failure for context ID {session_id}: {str(e)}")

# -----------------------------------------------------------------------------
# ROUTE ENDPOINTS
# -----------------------------------------------------------------------------
@app.post(
    "/sessions",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Establish a new live in-memory AI debate instance session context.",
)
async def create_debate_session(payload: SessionCreateRequest):
    session_id = str(uuid.uuid4())
    ai_position: Literal["FOR", "AGAINST"] = "AGAINST" if payload.position == "FOR" else "FOR"
    
    new_session = {
        "session_id": session_id,
        "topic": payload.topic,
        "user_position": payload.position,
        "ai_position": ai_position,
        "duration_minutes": payload.duration_minutes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ACTIVE",
        "conversation_history": [],
        # Milestone 2: Initialize analytics fields
        "analysis_status": "NOT_STARTED",
        "analysis_report": None
    }
    
    active_sessions[session_id] = new_session
    logger.info(f"Successfully generated new debate session channel record ID: {session_id}")
    
    return {
        "session_id": session_id,
        "ai_position": ai_position,
        "status": "ACTIVE"
    }


@app.post(
    "/sessions/{session_id}/opening",
    response_model=OpeningStatementResponse,
    summary="Synthesize an initial conversational argument stance frame from the LLM engine.",
)
async def generate_opening_statement(
    session_id: str = Path(..., description="The unique session identifier uuid string.")
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
    
    session = active_sessions[session_id]
    
    if session["status"] != "ACTIVE":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot generate opening items inside an inactive track.")
        
    if len(session["conversation_history"]) > 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Opening speech component has already been populated here.")

    ai_speech = await generate_ai_response(
        topic=session["topic"],
        ai_position=session["ai_position"],
        user_position=session["user_position"],
        conversation_history=[],
        is_opening=True
    )
    
    entry = {
        "speaker": "ai",
        "message": ai_speech,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    session["conversation_history"].append(entry)
    
    return {
        "response": ai_speech,
        "transcript_entry": entry
    }


@app.post(
    "/sessions/{session_id}/message",
    response_model=DebateTurnResponse,
    summary="Accept incoming user speech statements and return structured analytical rebuttals.",
)
async def live_debate_turn(
    session_id: str = Path(..., description="The unique target debate session token value."),
    payload: DebateMessageRequest = ... # type: ignore
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
        
    session = active_sessions[session_id]
    
    if session["status"] != "ACTIVE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="This specific debate track channel session has already completed or closed."
        )

    user_entry = {
        "speaker": "user",
        "message": payload.message.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    history_state = list(session["conversation_history"])
    
    ai_rebuttal = await generate_ai_response(
        topic=session["topic"],
        ai_position=session["ai_position"],
        user_position=session["user_position"],
        conversation_history=history_state,
        latest_user_message=payload.message.strip(),
        is_opening=False
    )
    
    session["conversation_history"].append(user_entry)
    
    ai_entry = {
        "speaker": "ai",
        "message": ai_rebuttal,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    session["conversation_history"].append(ai_entry)
    
    return {
        "response": ai_rebuttal,
        "history_length": len(session["conversation_history"])
    }


@app.get(
    "/sessions/{session_id}/transcript",
    response_model=TranscriptResponse,
    summary="Expose sequential ordered interaction history frames logged inside a targeted room session identifier.",
)
async def get_debate_transcript(
    session_id: str = Path(..., description="The unique debate identifier target token key.")
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
        
    return {
        "session_id": session_id,
        "messages": active_sessions[session_id]["conversation_history"]
    }


@app.get(
    "/sessions/{session_id}",
    response_model=SessionStateResponse,
    summary="Query current structural details and operation state statistics corresponding to an ongoing challenge trace tracker.",
)
async def get_session_state(
    session_id: str = Path(..., description="The unique tracking room instance signature target hash.")
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
        
    session = active_sessions[session_id]
    return {
        "topic": session["topic"],
        "user_position": session["user_position"],
        "ai_position": session["ai_position"],
        "status": session["status"],
        "duration_minutes": session["duration_minutes"],
        "message_count": len(session["conversation_history"])
    }


@app.post(
    "/sessions/{session_id}/end",
    response_model=EndDebateResponse,
    summary="Force termination operations across an open debate lane and trigger background evaluation analysis.",
)
async def terminate_debate_session(
    session_id: str = Path(..., description="The distinct session system routing token value.")
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
        
    session = active_sessions[session_id]
    session["status"] = "ENDED"
    session["ended_at"] = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Debate sequence session context identifier explicitly flagged as ENDED: {session_id}")
    
    # Milestone 2: Trigger background processing task if sufficient context has been established
    if len(session["conversation_history"]) >= 4:
        session["analysis_status"] = "PROCESSING"
        asyncio.create_task(run_background_analysis_task(session_id))
    else:
        logger.warning(f"Session {session_id} ended with insufficient messages ({len(session['conversation_history'])}). Background analysis bypassed.")
        session["analysis_status"] = "FAILED"

    return {"status": "ENDED"}


@app.get(
    "/sessions",
    response_model=List[SessionListItem],
    summary="Enumerate brief summaries mapping out active running system channels.",
)
async def list_active_sessions():
    result_list = []
    for s_id, data in active_sessions.items():
        if data["status"] == "ACTIVE":
            result_list.append({
                "session_id": s_id,
                "topic": data["topic"],
                "status": data["status"],
                "created_at": data["created_at"]
            })
    return result_list


# --- MILESTONE 2: NEW ANALYTICS ENDPOINTS ---

@app.post(
    "/sessions/{session_id}/analyze",
    response_model=DebateAnalysisReport,
    summary="Manually trigger or fetch a cached post-debate performance report assessment.",
)
async def analyze_debate_session(
    session_id: str = Path(..., description="Target session identity tracking code.")
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
        
    session = active_sessions[session_id]
    
    # Minimum transcript check rule enforcement (4 messages required)
    if len(session["conversation_history"]) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"The conversation history requires at least 4 statements to generate coaching data. Current count: {len(session['conversation_history'])}"
        )
        
    # Return cached content if generation has completed already
    if session.get("analysis_report") is not None:
        logger.info(f"Returning cached evaluation metrics report instance for room: {session_id}")
        return session["analysis_report"]
        
    session["analysis_status"] = "PROCESSING"
    try:
        report = await generate_debate_analysis(
            topic=session["topic"],
            user_position=session["user_position"],
            ai_position=session["ai_position"],
            transcript=session["conversation_history"]
        )
        session["analysis_report"] = report
        session["analysis_status"] = "COMPLETED"
        return report
    except Exception as err:
        session["analysis_status"] = "FAILED"
        raise err


@app.get(
    "/sessions/{session_id}/report",
    response_model=DebateAnalysisReport,
    summary="Fetch the pre-generated caching ledger tracking object mapping to this conversation instance room.",
)
async def get_debate_report(
    session_id: str = Path(..., description="Target session tracking hash code value.")
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
        
    session = active_sessions[session_id]
    
    if session.get("analysis_report") is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The performance evaluation evaluation report instance has not been compiled yet. Please check status endpoint hooks."
        )
        
    return session["analysis_report"]


@app.get(
    "/sessions/{session_id}/analysis-status",
    response_model=AnalysisStatusResponse,
    summary="Query out-of-band processing metrics to ascertain generation progression.",
)
async def get_analysis_status(
    session_id: str = Path(..., description="Target verification session signature target token.")
):
    if session_id not in active_sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target conversation session not tracked.")
        
    return {"status": active_sessions[session_id]["analysis_status"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)