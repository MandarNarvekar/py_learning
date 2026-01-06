from fastapi import FastAPI
app = FastAPI() 
@app.get("/Mandar")
def read_root():
    return {"Hello": "Mandar"}       