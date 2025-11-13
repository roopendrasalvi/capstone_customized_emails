from fastapi import FastAPI, Body
import uvicorn

app = FastAPI()

@app.get("/input-email")
async def input_email(
    email_subject: str = Body(..., embed=True),
    email_body: str = Body(..., embed=True)
):
    return {
        "subject": email_subject,
        "body": email_body
    }

if __name__ == "__main__":
   
    uvicorn.run("email_input:app", host="127.0.0.1", port=8051, reload=True)
