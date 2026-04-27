from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from models import ChatRequest, ChatResponse, FileUploadResponse
from parsers import parse_csv, parse_pdf
from vdb import store_transactions_in_vdb, clear_vdb
from ai_service import chat_with_granite

app = FastAPI(title="Personal Finance Assistant API")

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def serve_dashboard(request: Request):
    """
    Serves the main minimalistic UI dashboard.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running."}

@app.post("/clear")
def clear_data():
    """
    Endpoint to wipe the vector database entirely.
    """
    success = clear_vdb()
    if success:
        return {"status": "success", "message": "All transaction data has been deleted."}
    raise HTTPException(status_code=500, detail="Failed to clear vector database.")

@app.post("/upload/file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a bank statement.
    Parses the file, extracts transactions, and stores them in Qdrant as vectors.
    """
    content = await file.read()
    filename = file.filename.lower()
    
    transactions = []
    
    try:
        if filename.endswith(".csv"):
             transactions = parse_csv(content)
        elif filename.endswith(".pdf"):
             transactions = parse_pdf(content)
        else:
             raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or PDF.")
             
        if not transactions:
            return FileUploadResponse(status="success", message="File parsed, but no transactions found.", num_transactions=0)
            
        # Store securely to Vector DB
        store_transactions_in_vdb(transactions)
        
        return FileUploadResponse(
            status="success",
            message=f"Successfully extracted and stored {len(transactions)} transactions.",
            num_transactions=len(transactions)
        )
        
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint for users to ask questions about their statements.
    Ties together Granite 4, tool calling, and embedded search.
    """
    try:
        reply_text, updated_history = chat_with_granite(request.messages)
        return ChatResponse(reply=reply_text, history=updated_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
        
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=1578, reload=True)
