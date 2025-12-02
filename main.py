import os
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import whisper
import uvicorn
from app.utils.extract import extract_questions_answers
from app.agents.search_agent import web_search
from app.agents.compare import compare_answer_with_knowledge, llm_search
from app.utils.report import generate_report

print("main.py loaded successfully")

app = FastAPI()

# Load Whisper model once
model = whisper.load_model("small")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def convert_to_wav(input_path: str) -> str:
    """Convert audio to WAV using ffmpeg."""
    output_path = os.path.splitext(input_path)[0] + ".wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, output_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output_path


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    file_path = None
    wav_path = None

    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert to WAV
        if file.filename.lower().endswith(".wav"):
            wav_path = file_path
        else:
            wav_path = convert_to_wav(file_path)

        # Run whisper
        result = model.transcribe(wav_path)
        transcript = result["text"]
        print("TRANSCRIPT:", transcript)

        # Extract Questions + Answers
        qa_list = extract_questions_answers(transcript)
        print("EXTRACTED QA:", qa_list)
        for qa in qa_list:
            question = qa["question"]
            candidate_answer = qa["candidate_answer"]

            # Optional: get knowledge summary
            knowledge_summary = llm_search(question)
            qa["knowledge_summary"] = knowledge_summary

            # Optional: compare candidate answer with knowledge
            comparison = compare_answer_with_knowledge(question, candidate_answer)
            qa["correctness"] = comparison["correctness"]
            qa["completeness"] = comparison["completeness"]
            qa["explanation"] = comparison["explanation"]

        return JSONResponse({
            "transcript": transcript,
            "qa_extracted": qa_list
        })
        
        

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    finally:
        # cleanup
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        if wav_path and os.path.exists(wav_path) and wav_path != file_path:
            os.remove(wav_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)